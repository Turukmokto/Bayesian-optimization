import math
import numbers
import random

from math import log
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from scipy.io.arff import loadarff


def arff_load(name):
    raw_data = loadarff(name)
    df_data = pd.DataFrame(raw_data[0])
    str_df = df_data.select_dtypes([object])
    str_df = str_df.stack().str.decode('utf-8').unstack()
    for col in str_df:
        df_data[col] = str_df[col]
    df_data, relation_cat = category_coding(df_data)
    return df_data


def category_coding(df):
    cat_col = df.select_dtypes(include=[object]).axes[1]
    num_col = df.select_dtypes(include=[np.number]).axes[1]
    df_new = df.copy()
    len_df = len(df)
    values = {}
    flag = 0
    for col in cat_col:
        num_miss = df_new[col].isnull().sum()
        if num_miss / len_df >= 0.8:
            df_new = df_new.drop(columns=col, axis=1)
        elif 0.8 > num_miss / len_df >= 0.1:
            values[col] = 'no'
            flag = 1
    for col in num_col:
        num_miss = df_new[col].isnull().sum()
        if num_miss / len_df > 0.5:
            df_new = df_new.drop(columns=col, axis=1)
    if flag == 1:
        df_new.fillna(value=values, inplace=True)
    cat_col = df_new.select_dtypes(include=[object]).axes[1]
    num_col = df_new.select_dtypes(include=[np.number]).axes[1]
    for col in cat_col:
        df_new[col] = pd.factorize(df_new[col])[0]
        df_new.loc[(df_new[col] == -1), col] = df_new[col].median()
    for col in num_col:
        values[col] = df_new[col].median()
    df_new.fillna(value=values, inplace=True)
    relation = round(len(cat_col) / len(df_new.axes[1]), 2)
    return df_new, relation


def get_key(val, my_dict):
    for key, value in my_dict.items():
        if val == value:
            return key


def honest_count(params, df):
    if isinstance(params['criterion'], numbers.Number):
        params['criterion'] = get_key(params['criterion'], {'gini': 0, 'entropy': 1})
    name_pred = df.columns[-1]
    y_train_full = df[name_pred]
    x_train_full = df.drop(name_pred, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(x_train_full, y_train_full, train_size=0.8, random_state=42)
    model = DecisionTreeClassifier()
    model.set_params(**params)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    try:
        tree_score = f1_score(y_test, pred)
    except ValueError:
        tree_score = f1_score(y_test, pred, average='micro')
    return tree_score


def surrogate_fun(df_surrogate):
    name_pred = df_surrogate.columns[-1]
    y_train_full = df_surrogate[name_pred]
    x_train_full = df_surrogate.drop(name_pred, axis=1)
    model = RandomForestRegressor(n_estimators=100)
    model.fit(x_train_full, y_train_full)
    return model


def initialization(primary_df, df_surrogate, val_start):
    for i in range(val_start):
        number = random.randint(0, len(primary_df) - 1)
        df_surrogate = df_surrogate.append(primary_df.iloc[number], ignore_index=True)
    return df_surrogate


def enhancement_function(df_surrogate, primary_df, lamda):
    results = []
    list_trees = surrogate_fun(df_surrogate).estimators_
    for model_tree in list_trees:
        results.append(model_tree.predict(primary_df))
    means = np.average(results, axis=0)
    stds = np.std(results, axis=0)
    lamda_stds = [i * lamda for i in stds]
    return list(map(sum, zip(means, lamda_stds)))


def for_real(data, df_surrogate):
    index_nan = pd.isnull(df_surrogate).any(1).to_numpy().nonzero()[0]
    name_pred = df_surrogate.columns[-1]
    while df_surrogate['f1'].isnull().sum() != 0:
        for index in index_nan:
            params = df_surrogate.drop(name_pred, axis=1).iloc[index]
            score = honest_count(params, data)
            df_surrogate.at[index, 'f1'] = score
    return df_surrogate


def lamda_up(step, lamda):
    return lamda * math.isqrt(int(log(step + 1)))


def random_search(data, num_step, df_random):
    for i in range(0, num_step):
        number = random.randint(0, len(data) - 1)
        df_random = df_random.append(data.iloc[number], ignore_index=True)
    return df_random
