"""
This file is not imported and used in our training process.
"""
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from utils import fetch_train_data, evaluate_model


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
        self.bert = AutoModel.from_pretrained(language_model)
        hidden_size = self.bert.config.hidden_size
        self.fc = nn.Linear(hidden_size, 3)

    def forward(self, x):
        x = x.to(self.device)
        encode = self.bert(x)[0][:, 0, :]
        result = self.fc(encode)
        return result


class TuneModel(nn.Module):
    def __init__(self, device: str, model_path='../data/model.pt'):
        super(TuneModel, self).__init__()
        checkpoint = torch.load(model_path)
        model = FitModel(device=device)
        model.load_state_dict(checkpoint['model'])
        self.device = device
        self.bert = model.bert
        hidden_size = self.bert.config.hidden_size
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 2)
        )

    def forward(self, x):
        with torch.no_grad():
            encode = self.bert(x)[0][:, 0, :]
        return self.fc(encode)


def padder(batch):
    def partial_padding(encode, max_length):
        return [encode[i] if i < len(encode) else 0 for i in range(max_length)]

    if len(batch) != 2:
        encodes = batch
        dim = max([len(encode) for encode in encodes])
        encodes = [partial_padding(encode, dim) for encode in encodes]
        return torch.LongTensor(encodes)

    encodes, label = zip(*batch)
    dim = max([len(encode) for encode in encodes])
    encodes = [partial_padding(encode, dim) for encode in encodes]
    return torch.LongTensor(encodes), torch.LongTensor(label)


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
    predictions = list()
    ground_truth = list()
    for batch in valid_iter:
        x, y = batch
        logits = model(x)
        _, preds = torch.max(logits.data, dim=1)
        predictions += preds.to('cpu').tolist() 
        ground_truth += y.to('cpu').tolist()
    return evaluate_model(ground_truth, predictions)


def train(trainset, validset, batch_size=16, lr=3e-5, n_epochs=10, load=False, tune=False):
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
    optimizer = AdamW(model.parameters(), lr=lr)

    num_steps = (len(trainset) // batch_size) * n_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_steps)

    if tune:
        model = TuneModel(device, model_path)
    else:
        model = FitModel(device=device)

    model_path = "../data/model.pt"
    tuned_model_path = "../data/tuned_model.pt"
    if load:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    model = model.to(device)
    for epoch in range(n_epochs):
        model.train()
        loss = train_step(train_iter, model, optimizer, scheduler)
        print(f'epoch: {epoch}, loss: {loss}')
        model.eval()
        print(evaluate_step(valid_iter, model))
        checkpoint = {'model': model.state_dict(), 
              'optimizer': optimizer.state_dict(), 
              'scheduler': scheduler.state_dict(),
              'epoch': epoch}
        if tune:
            torch.save(checkpoint, tuned_model_path)
        torch.save(checkpoint, model_path)
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
    df = df[['fit']]
    labels = [row.to_list()[0] for _, row in df.iterrows()]
    labels = [1 if fit.startswith('S') else 2 if fit.startswith('T') else 3 for fit in labels]
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
    length = len(labels)
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
            predictions += batch_preds.to('cpu').tolist()
    print(predictions)

if __name__ == '__main__':
    main()
