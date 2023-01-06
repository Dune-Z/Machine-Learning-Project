import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from utils import fetch_train_data


class FitDataset(Dataset):
    def __init__(self, review_strings, fit_strings=None, max_length=256):
        super(FitDataset, self).__init__()
        self.max_length = max_length
        self.review_strings = review_strings
        self.fit_strings = fit_strings
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')

    def __len__(self):
        return len(self.fit_strings)

    def __getitem__(self, index):
        x = self.tokenizer.encode(
            self.review_strings[index], 
            max_length=self.max_length, 
            truncation=True)
        if(self.fit_strings==None):
            return x
        return x, self.fit_strings[index]


class FitModel(nn.Module):
    def __init__(self, device: str, language_model='roberta-base'):
        super(FitModel, self).__init__()
        self.device = device
        self.model = AutoModel.from_pretrained(language_model)
        hidden_size = self.model.config.hidden_size
        self.fc = nn.Linear(hidden_size, 3)

    def forward(self, x):
        x = x.to(self.device)
        encode = self.model(x)[0][:, 0, :]
        result = self.fc(encode)
        return result


def padder(batch):
    if len(batch) == 3:
        x1, x2, y = zip(*batch)
        max_length = max([len(x) for x in x1 + x2])
        x1 = [xi + [0] * (max_length - len(xi)) for xi in x1]
        x2 = [xi + [0] * (max_length - len(xi)) for xi in x2]
        return torch.LongTensor(x1), torch.LongTensor(x2), torch.LongTensor(y)
    else:
        x, y = zip(*batch)
        max_length = max([len(k) for k in x])
        x = [xi + [0] * (max_length - len(xi)) for xi in x]
        return torch.LongTensor(x), torch.LongTensor(y)


def train_step(train_iter, model, optimizer, scheduler):
    criterion = nn.CrossEntropyLoss()
    mean_loss = 0
    for i, batch in enumerate(train_iter):
        optimizer.zero_grad()
        x, y = batch
        prediction = model(x)
        loss = criterion(prediction, y.to(model.device))
        loss.backward()
        optimizer.step()
        scheduler.step()
        mean_loss = mean_loss * i + loss
        mean_loss /= (i + 1)
        if i % 100:
            print(f'>>step {i}, loss: {loss}')
        del loss
    return mean_loss


def evaluate_step(valid_iter, model):
    total = correct = 0
    for batch in valid_iter:
        x, y = batch
        logits = model(x)
        _, predicted = torch.max(logits.data, dim=1)
        total += y.size(0)
        correct += (predicted.to(model.device) == y.to(model.device)).sum().item()
    print(f'validation accuracy: {(100 * correct / total)}%')


def train(trainset, validset, batch_size=16, lr=3e-5, n_epochs=10, load=False):
    train_iter = DataLoader(dataset=trainset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=0,
                            collate_fn=padder)
    valid_iter = DataLoader(dataset=validset,
                            batch_size=batch_size*16,
                            shuffle=False,
                            num_workers=0,
                            collate_fn=padder)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = FitModel(device=device)
    optimizer = AdamW(model.parameters(), lr=lr)

    num_steps = (len(trainset) // batch_size) * n_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_steps)
    if load:
        checkpoint = torch.load('../data/model.pt')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
    model = model.to(device)
    for epoch in range(n_epochs):
        model.train()
        loss = train_step(train_iter, model, optimizer, scheduler)
        print(f'epoch: {epoch}, loss: {loss}')
        model.eval()
        evaluate_step(valid_iter, model)
        checkpoint = {'model': model.state_dict(), 
              'optimizer': optimizer.state_dict(), 
              'scheduler': scheduler.state_dict(),
              'epoch': epoch}
        torch.save(checkpoint, '../data/model.pt')
    return model


def make_dataset(df: pd.DataFrame):
    X = list()
    x_col = ['review_summary', 'review', 'rating']
    df = df[x_col]
    for _, rows in df.iterrows():
        contents = rows.to_list()
        contents = ['[CLS] ' + col + ' [SEP] ' + content + '[SEP]'
        if isinstance(content, str) else '[CLS] ' + col + ' [SEP] None [SEP]' for col, content in zip(x_col, contents)]
        X.append("".join(contents))
    return X


def make_label(df: pd.DataFrame):
    df = df['fit']
    labels = [row.to_list()[0] for _, row in df.iterrows()]
    labels = [0 if fit.startswith('S') else 1 if fit.startswith('T') else 2 for fit in labels]
    return labels


def main():
    df = fetch_train_data()
    review_col = ['review_summary', 'review', 'rating', 'fit']
    review_df = df[review_col]
    train_df = review_df.loc[~review_df['fit'].isna()]
    test_df = review_df.loc[review_df['fit'].isna()]

    data = make_dataset(train_df)
    labels = make_label(train_df)
    # train valid split
    length = len()
    indexes = random.sample(range(0, length), length)
    train_data = [data[index] for index in indexes[length // 5: ]]
    train_label = [labels[index] for index in indexes[length // 5: ]]
    valid_data = [data[index] for index in indexes[:length // 5]]
    valid_label = [labels[index] for index in indexes[:length // 5]]

    trainset = FitDataset(train_data, train_label)
    validset = FitDataset(valid_data, valid_label)
    model = train(trainset, validset)

    test_data = make_dataset(test_df)
    testset = FitDataset(test_data)
    test_iter = DataLoader(testset, collate_fn=padder)
    predictions = list()
    with torch.no_grad():
        model.eval()
        for batch in test_iter:
            logits = model(batch)
            _, batch_preds = torch.max(logits.data, dim=1)
            predictions.append(batch_preds.to('cpu'))
    return predictions

if __name__ == '__main__':
    main()
