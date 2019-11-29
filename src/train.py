import data_import
import os
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, learning_curve

def outlier_handler(train, feature_name):
    """
    Truncate outlier by 0.05 of data as lower bound and
    0.95 of data as upper bound.

    Args:
        train: data object.
        feature_name: feature name (e.g. 'booking_bool').

    Returns:
        None.
    """
    ub = train[feature_name].quantile(.95)
    lb = train[feature_name].quantile(.05)
    train.loc[train[feature_name] > ub, feature_name] = ub
    train.loc[train[feature_name] < lb, feature_name] = lb

def impute_with_best_support(train, feature_name, feature_support_name):
    """
    Impute the feature with with its most correlated feature.

    Args:
        train: data object.
        feature_name: target feature name with nan.
        feature_support_name: most correlated feature

    Returns:
        None.
    """
    feature_support_avg = train[feature_name].mean()
    feature_support_std = train['prop_location_score1'].std()
    feature_support_null_count = train[feature_support_name].isnull().sum()
    if train[feature_name].dtypes == np.float64:
        feature_support_null_random_list = np.random.uniform(feature_support_avg - feature_support_std, feature_support_avg + feature_support_std, 1)
    if train[feature_name].dtypes == np.int64:
        feature_support_null_random_list = np.random.randint(feature_support_avg - feature_support_std, feature_support_avg + feature_support_std, 1)
    train.loc[np.isnan(train[feature_support_name]), feature_support_name] = feature_support_null_random_list

def pop_dest_finder(train, feature_name):
    """
    Find the most popular destinations and make it as boolean.
    """
    df = train.groupby([feature_name])["click_bool"].mean().sort_values(ascending=False).to_frame().reset_index()
    df['popular_bool'] = pd.cut(df["click_bool"], df['click_bool'].quantile([.0, .75, 1]), labels=['not popular', 'popular'])
    popular_destinations = df.loc[df['popular_bool']=='popular', 'srch_destination_id']
    train['popular_destination_bool'] = 0
    train.loc[train['srch_destination_id'].isin(popular_destinations) , 'popular_destination_bool'] = 1

def check_nan_values(train):
    """
    Check if the data has nan.
    """
    nan_value_number = train.isnull().values.sum()
    if nan_value_number == 0:
      print("Success: there is no nan value")
    else:
      print("Error: there are {} nan values".format(nan_value_number))

def feature_eng(train):
    """
    Feature engineering for the data set.
    """
    ## impute 'prop_review_score' with its median
    train['prop_review_score'].isnull().sum()
    train['prop_review_score'] = train['prop_review_score'].fillna(train['prop_review_score'].median())

    ## impute 'prop_location_score2' with 0
    train['prop_location_score2'].fillna(0, inplace=True)

    ## impute 'visitor_hist_adr_usd' with 0
    train['visitor_hist_adr_usd'].fillna(0, inplace=True)

    ## place dummy feature for 'visitor_hist_starrating' presence
    train['visitor_hist_starrating_bool'] = pd.notnull(train['visitor_hist_starrating']) * 1

    ## extract month, day, hour, minute, dayofweek, quarter from 'date_time'
    train['date_time'] = pd.to_datetime(train['date_time'])
    for prop in ["month", "day", "hour", "minute", "dayofweek", "quarter"]:
        train[prop] = getattr(train["date_time"].dt, prop)

    ## smooth 'prop_log_historical_price' with 1
    train.loc[train['prop_log_historical_price']!=0, 'prop_log_historical_price'] = 1

    ## find the popular destinations
    train['visitor_location_country_bool'] = np.where(train['visitor_location_country_id']==train['visitor_location_country_id'].value_counts().index[:2][0], 1, 0)

    ## impute 'srch_query_affinity_score' with its best support (most correlated feature) and censor the outlier
    impute_with_best_support(train, 'srch_query_affinity_score', \
        train.drop(['date_time'], axis=1).astype(float).corr()['srch_query_affinity_score'].abs().sort_values(ascending = False).index[0]
        )
    outlier_handler(train, 'srch_query_affinity_score')

    ## impute 'orig_destination_distance' with its best support
    impute_with_best_support(train, 'prop_location_score1', \
        train.drop(['date_time'], axis=1).astype(float).corr()['orig_destination_distance'].abs().sort_values(ascending = False).index[0]
        )

    ##  merge 9 competitors' price info
    for i in range(1,9):
        train['comp'+str(i)+'_rate'].fillna(0, inplace=True)
    train['comp_rate_sum'] = train['comp1_rate']
    for i in range(2,9):
        train['comp_rate_sum'] += train['comp'+str(i)+'_rate']

    for i in range(1,9):
        train['comp'+str(i)+'_inv'].fillna(0, inplace=True)
        train.loc[train['comp'+str(i)+'_inv']==1, 'comp'+str(i)+'_inv'] = 10
        train.loc[train['comp'+str(i)+'_inv']==-1, 'comp'+str(i)+'_inv'] = 1
        train.loc[train['comp'+str(i)+'_inv']==0, 'comp'+str(i)+'_inv'] = -1
        train.loc[train['comp'+str(i)+'_inv']==10, 'comp'+str(i)+'_inv'] = 0
    train['comp_inv_sum'] = train['comp1_inv']
    for i in range(2,9):
        train['comp_inv_sum'] += train['comp'+str(i)+'_inv']

def get_features(train):
    """
    Extract the features that will be used for training.
    """
    feature_names = list(train.columns)
    feature_names.remove('date_time')
    feature_names.remove('site_id')
    feature_names.remove('visitor_location_country_id')
    feature_names.remove('prop_country_id')
    feature_names.remove('srch_destination_id')
    feature_names.remove("visitor_hist_starrating")

    for i in range(1,9):
        feature_names.remove('comp'+str(i)+'_rate')
        feature_names.remove('comp'+str(i)+'_inv')
        feature_names.remove('comp'+str(i)+'_rate_percent_diff')

    if "position" in feature_names:
        ## only true in the training set
        feature_names.remove("position")
    if "gross_bookings_usd" in feature_names:
        ## only true in the training set
        feature_names.remove("gross_bookings_usd")
    if "click_bool" in feature_names:
        ## only true in the training set
        feature_names.remove("click_bool")
    if "booking_bool" in feature_names:
        ## only true in the training set
        feature_names.remove("booking_bool")

    return feature_names

def main():
    train = data_import.load_train()
    feature_eng(train)
    train_set = train.sample(n=1000)

    ## Train the booking model
    for i in range(0,2):

        if i==0:
            model_name = "Booking"
            outcome_name = "booking_bool"
            isBook = True

        else:
            model_name = "Click"
            outcome_name = "click_bool"
            isBook = False

        ## downsampling
        data_trainset = train_set[train_set[outcome_name]==1]
        data_rows = data_trainset.index.tolist()
        data_trainset = data_trainset.append(train_set.ix[random.sample(train_set.drop(data_rows).index, len(data_rows))])
        train_sample = data_trainset

        print("Training the {} Classifier...".format(model_name))
        tstart = datetime.now()
        feature_names = get_features(train_sample)
        print("Using {} features ...".format(len(feature_names)))
        X = train_sample[feature_names].values
        Y = train_sample[outcome_name].values

        ## Randomized Parameter Optimization on classifier N0.1: RandomForestClassifier
        param_grid = {"min_samples_leaf": range(1, 10),
            "max_depth": [2, 8],
            "n_estimators": range(1, 100)}
        research = RandomizedSearchCV(estimator=RandomForestClassifier(),
            param_distributions=param_grid, scoring='accuracy', cv=3)
        research.fit(X, Y)

        rf_est = RandomForestClassifier(
            max_features='sqrt', min_samples_split=4, verbose=1,
            criterion='gini', n_jobs=50, random_state=42,
            max_depth=research.best_params_['max_depth'],
            min_samples_leaf=research.best_params_['min_samples_leaf'],
            n_estimators=research.best_params_['n_estimators']
            )

        ## Randomized Parameter Optimization on classifier N0.2: GradientBoostingClassifier
        param_grid = {"min_samples_leaf": range(1, 10),
            "max_depth": [2, 8],
            "n_estimators": range(1, 1000)}
        research = RandomizedSearchCV(estimator=GradientBoostingClassifier(),
            param_distributions=param_grid, scoring='accuracy', cv=3)
        research.fit(X, Y)

        gbm_est = GradientBoostingClassifier(
            learning_rate=0.0008, loss='exponential', min_samples_split=3, max_features='sqrt',random_state=42, verbose=1,
            max_depth=research.best_params_['max_depth'],
            min_samples_leaf=research.best_params_['min_samples_leaf'],
            n_estimators=research.best_params_['n_estimators']
            )

        ## Randomized Parameter Optimization on classifier N0.3: ExtraTreesClassifier
        param_grid = {"n_jobs": range(1, 100),
            "max_depth": [2, 8],
            "n_estimators": range(1, 100)}
        research = RandomizedSearchCV(estimator=ExtraTreesClassifier(),
            param_distributions=param_grid, scoring='accuracy', cv=3)
        research.fit(X, Y)

        et_est = ExtraTreesClassifier(
            max_features='sqrt', criterion='entropy',
            random_state=42,  verbose=1,
            max_depth=research.best_params_['max_depth'],
            n_jobs=research.best_params_['n_jobs'],
            n_estimators=research.best_params_['n_estimators']
            )
        voting_est =VotingClassifier(
            estimators=[('rf', rf_est),('gbm', gbm_est),('et', et_est)],
            voting='soft', weights=[3,5,2], n_jobs=50)

        voting_est.fit(X,Y)

        ## Save classifier
        print("Saving the classifier...")
        tstart = datetime.now()
        data_import.save_model(voting_est, isBook)
        print("Time used:" + str(datetime.now() - tstart) + "\n")


if __name__=="__main__":
    main()
