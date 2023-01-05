# Machine Learning Project

## File tree

```
.
├── data                        # Auto-downloaded datasets
│   └── ...
├── notebooks                   # Your notebooks with experimental results
│   └── ...
├── models                      # Trained models (probably saved with pickle)
│   └── ...
├── report                      # Project report (including source files, figures, etc.)
│   └── ...
├── scripts                     # Utility scripts (download, deploy, etc.)
│   └── ...
└── src                         # Source files (including train.py & test.py)
    └── ...
```

### Description of each file

| File                   | Description                                                                                                      |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `data/train_data.json` | The original dataset auto-downloaded by `fetch_train_data()` in `src/utils.py`                                   |
| `scripts/download.py`  | Script to download all datasets (MiraLab's version, UCSD's version)                                              |
| `src/train.py`         | Script to train our model (not necessary, since we may finish training all in notebooks)                         |
| `src/test.py`          | Script to test our model (needed to generate submission file)                                                    |
| `src/models.py`        | Implementation of our model(s), should be wrapped as a class                                                     |
| `src/preprocess.py`    | Source code for data cleansing, transforming, and feature engineering                                            |
| `src/utils.py`         | Utility functions that cannot be put in other source files (e.g. `fetch_train_data()`)                           |
| `src/main.ipynb`       | An example notebook to show how to import and use our library. You may have your own versions in other branches. |

## TODO

-   [ ] Implement an oversampling method to deal with the imbalanced dataset

## Reference

Please share your references here.

| Title                                                                                                                                                                                                                                                                                                                | Comment             |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------- |
| _BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers) (Minneapolis, Minnesota, Jun. 2019), 4171–4186._ | Refered in homepage |
| _Convolutional Neural Networks for Sentence Classification. Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP) (Doha, Qatar, Oct. 2014), 1746–1751._                                                                                                                     | Refered in homepage |
| _Recommending Product Sizes to Customers. Proceedings of the Eleventh ACM Conference on Recommender Systems (New York, NY, USA, Aug. 2017), 243–250._                                                                                                                                                                |                     |
| _Decomposing fit semantics for product size recommendation in metric spaces. Proceedings of the 12th ACM Conference on Recommender Systems (New York, NY, USA, Sep. 2018), 422–426._                                                                                                                                 | Original dataset    |
| _Learning Embeddings for Product Size Recommendations. (2019), 9._                                                                                                                                                                                                                                                   |                     |
| _PreSizE: Predicting Size in E-Commerce using Transformers. Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval (New York, NY, USA, Jul. 2021), 255–264._                                                                                                |                     |
