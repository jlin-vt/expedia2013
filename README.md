## Introduction
This repository contains `python` code for the Kaggle competition: [Personalize Expedia Hotel Searches - ICDM 2013](https://www.kaggle.com/c/expedia-personalized-sort). The goal of this challenge is to predict which hotels a consumer behavior (click on or/and a purchase a hotel) and to sort the hotels so that these properties appear first in the search results.

### Data
The data provided by the popular travel website [Expedia.com](https://www.expedia.com/) contains two `.csv` files: `train.csv` and `test.csv`. They provide datat that contains:

1. Hotel characteristics
2. Location attractiveness of hotels
3. User’s aggregate purchase history
4. Competitive OTA information

In the `train.csv`, there are roughly 10 million records (400 thousand searches) and 54 features. This challenge is a typical classification problems with imbalanced data, as about 4% of searches lead to click/booking behavior.

### Model
We consider a point-wise [LeToR (Learning to rank)](https://en.wikipedia.org/wiki/Learning_to_rank) algorithm by ensembling multiple base classifiers.

### Evaluation
Models are evaluated with [nDCG (normalized Discounted Cumulative Gain)](https://www.kaggle.com/c/expedia-personalized-sort#evaluation).

### Result
The final submission was a mean ensemble of 35 best leaderboard submissions. The demo is able to score 0.51 nDCG on validation set, which is good enough to get top 10  out of 337 entries.

## Usage
To run the code, follow the steps below:

* `git clone` this repository to the local.
* Download the data from [Kaggle page](https://www.kaggle.com/c/expedia-personalized-sort/data) and unzip `.zip` files to folder `./data`.
* The tree of `expedia2013` folder should be like

```
├── R
│   ├── config.json
│   ├── data_import.R
│   ├── eda.R
│   ├── evalulation.R
│   ├── feature_engineering.R
│   ├── model.R
│   └── run.R
├── README.md
├── data
├── models
│   ├── book_model.pickle
│   ├── book_model.rda
│   ├── click_model.pickle
│   └── click_model.rda
├── python
│   ├── SETTINGS.json
│   ├── data_import.py
│   ├── eda.py
│   ├── predict.py
│   └── train.py
└── results
```

* Run `python ./src/python/train.py` to load data, generate features and train models.
* Run `python ./src/python/predict.py` to generate submission.
* Models and submissions are located in `./models` and `./results` respectively.

Note:

* The instruction is tested on an `AWS` `g2.2xlarge` instance running `ubuntu 14.0`.
* To change the file paths, you need to modify `./python/SETTINGS.json`.
* While this repository is written in `python`, a short `R` demo is also available. The `R` code runs on the small subset of the data as `R` may have troubles with large-scale data.

### Dependencies
* The `python` code requires `Python 2.7.0` , `numpy`,
`scikit-learn`, `pandas`.
* To run the `R` demo, you should install a few packages:
`ggplot2`, `dplyr`, `lubridate`, `rjson`, `h2o`.

### Documentation
See [my blog post](https://jlin-vt.github.io/posts/learning-personalized-hotel-searches-2/) for detailed description.

## Reference
- [Combination of Diverse Ranking Models for Personalized Expedia Hotel Searches](https://arxiv.org/pdf/1311.7679v1.pdf)
- [Expedia Recommendation System](https://github.com/shawnhero/ICDM2013-Expedia-Recommendation-System)
- [Information Retrieval - Stanford University](https://web.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf)
