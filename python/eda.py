import data_import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

def plot_discrete_1d (train, feature_name):
    """
    Make a countplot for a discrete feature.

    Args:
        train: data object.
        feature_name: feature name (e.g. 'booking_bool').

    Returns:
        None.
    """
    g = sns.countplot(x=feature_name, data=train)
    plt.show()

def plot_continous_1d (train, feature_name):
    """
    Make a boxplot for a continous feature.

    Args:
        train: data object.
        feature_name: feature name.

    Returns:
        None.
    """
    g = sns.boxplot(x=feature_name, data=train, orient="v")
    plt.show()

def plot_discrete_2d (train, feature_name):
    """
    Make two barplots for a discrete feature againt 'click_bool' and 'booking_bool'.

    Args:
        train: data object.
        feature_name: feature name.

    Returns:
        None.
    """
    fig,(ax1,ax2) = plt.subplots(ncols=2)
    fig.set_size_inches(20,8)
    sns.barplot(data=train,x=feature_name,y="click_bool",ax=ax1)
    sns.barplot(data=train,x=feature_name,y="booking_bool",ax=ax2)
    ax1.set(xlabel=feature_name, ylabel='click_bool')
    ax2.set(xlabel=feature_name, ylabel='booking_bool')
    plt.show()

def plot_continous_2d (train, feature_name):
    """
    Make two overlapping density plots for a continous feature againt 'click_bool' and 'booking_bool'.

    Args:
        train: data object.
        feature_name: feature name.

    Returns:
        None.
    """
    fig,(ax1,ax2) = plt.subplots(ncols=2)
    fig.set_size_inches(20, 8)
    sns.distplot(train[feature_name][(train["click_bool"]==0) & (train[feature_name].notnull())], color="Red", ax=ax1)
    sns.distplot(train[feature_name][(train["click_bool"]==1) & (train[feature_name].notnull())], color="Blue", ax=ax1)
    sns.distplot(train[feature_name][(train["booking_bool"]==0) & (train[feature_name].notnull())], color="Red", ax=ax2)
    sns.distplot(train[feature_name][(train["booking_bool"]==1) & (train[feature_name].notnull())], color="Blue", ax=ax2)
    ax1.set(xlabel=feature_name, ylabel='Frequency')
    ax2.set(xlabel=feature_name, ylabel='Frequency')
    ax1.legend(["Not Clicked", "Clicked"])
    ax2.legend(["Not Booked", "Booked"])
    plt.show()

def nan_percent(train, feature_name):
    """
    Make two barplots to see if the missing value is random in terms of 'click_bool' and 'booking_bool'.

    Args:
        train: data object.
        feature_name: feature name.

    Returns:
        None.
    """
    feature_nan_name = feature_name + "_na"
    train[feature_nan_name] = np.where(pd.isnull(train[feature_name]), 1, 0)
    fig,(ax1,ax2) = plt.subplots(ncols=2)
    fig.set_size_inches(20,8)
    sns.barplot(data=train,x=feature_nan_name,y="click_bool",ax=ax1)
    sns.barplot(data=train,x=feature_nan_name,y="booking_bool",ax=ax2)
    ax1.set(xlabel=feature_nan_name, ylabel='click_bool')
    ax2.set(xlabel=feature_nan_name, ylabel='booking_bools')
    plt.show()
    train.drop(labels = [feature_nan_name], axis = 1, inplace = True)

## Load data
train = data_import.load_train()
test = data_import.load_test()

## features count across datatype
dataTypeDf = pd.DataFrame(train.dtypes.value_counts()).reset_index().rename(columns={"index":"variableType",0:"count"})
fig,ax = plt.subplots()
fig.set_size_inches(20,5)
sns.barplot(data=dataTypeDf,x="variableType",y="count",ax=ax,color="#34495e")
ax.set(xlabel='Variable Type', ylabel='Count',title="Variables Count Across Datatype")
plt.show()

## feature completeness
missingValueColumns = train.columns[train.isnull().any()].tolist()
msno.bar(train[missingValueColumns], figsize=(20, 8),color="#34495e",fontsize=12,labels=True,)
msno.matrix(train[missingValueColumns],width_ratios=(10, 1), figsize=(20, 8),color=(0, 0, 0),fontsize=12,sparkline=True,labels=True)

## relatioships between two outcomes by crosstab
pd.crosstab(train['click_bool'], train['booking_bool'])

## how many hotels exist in both training and test data set
len(set(train['prop_id'].unique()))
len(set(test['prop_id'].unique()))
len(set(test['prop_id'].unique()) & set(train['prop_id'].unique()))

## EDA on 'prop_review_score' by various ways
plot_discrete_1d(train, "prop_review_score")
plot_discrete_2d(train, "prop_review_score")

train[['prop_review_score', 'booking_bool']].groupby(['prop_review_score'], as_index=False).mean().sort_values(by='prop_review_score', ascending=False)

train[['prop_review_score', 'booking_bool']].groupby(['prop_review_score']).mean().plot.bar()
plt.show()

sns.countplot('prop_review_score',hue='booking_bool',data=train)
plt.title('prop_review_score and booking_bool')
plt.show()

## EDA on 'prop_location_score2' by various ways
plot_continous_1d(train, "prop_location_score2")
plot_continous_2d(train, "prop_location_score2")

## 'prop_location_score2' correlation with other features
train.drop(['date_time'], axis=1).astype(float).corr()['prop_location_score2'].abs().sort_values(ascending=False)[1:10]

## 'prop_location_score2' correlation map under booking vs non-booking behavior
train[train['booking_bool']==1].drop(['date_time'], axis=1).astype(float).corr()['prop_location_score2'].abs().sort_values(ascending=False)[1:10]
train[train['booking_bool']==0].drop(['date_time'], axis=1).astype(float).corr()['prop_location_score2'].abs().sort_values(ascending=False)[1:10]

## correlation matrix for all features except 'date_time'
g = sns.heatmap(train.drop(['date_time'], axis=1).astype(float).corr(),annot=True, fmt = ".2f", cmap = "YlGnBu")
g.set_yticklabels(g.get_yticklabels(), rotation=0, fontsize=5)
g.set_xticklabels(g.get_xticklabels(), rotation=270, fontsize=5)
plt.show()
