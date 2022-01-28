from collections import Counter

import numpy as np
import pandas as pd
from random import seed
from random import random

from sklearn.model_selection import train_test_split


def select_only_cd1_1_test():
    metadata_test = pd.read_csv('../data/example_result_test.txt', delimiter=",")
    size = metadata_test.shape[0]
    iter = 0
    df = pd.DataFrame(columns=metadata_test.columns.values)
    for index, data in metadata_test.iterrows():
        iter = iter + 1
        if iter % 500 == 0:
            print(f"{(iter / size) * 100}%")
        if pd.notnull(data['genre_cd1_1']):
            df = df.append(data)

    df.to_csv('data/results_cd1_1_test.txt', index=False, sep=",")


def select_only_cd2_1_test():
    metadata_test = pd.read_csv('../data/example_result_test.txt', delimiter=",")
    size = metadata_test.shape[0]
    iter = 0
    df = pd.DataFrame(columns=metadata_test.columns.values)
    for index, data in metadata_test.iterrows():
        iter = iter + 1
        if iter % 500 == 0:
            print(f"{(iter / size) * 100}%")
        if pd.notnull(data['genre_cd2_1']):
            df = df.append(data)

    df.to_csv('data/results_cd2_1_rock20_train.txt', index=False, sep=",")


def select_only_cd1_1_train():
    metadata_train = pd.read_csv('../data/example_result_train.txt', delimiter=",")
    size = metadata_train.shape[0]
    iter = 0
    df = pd.DataFrame(columns=metadata_train.columns.values)
    for index, data in metadata_train.iterrows():
        iter = iter + 1
        if iter % 500 == 0:
            print(f"{(iter / size) * 100}%")
        if pd.notnull(data['genre_cd1_1']):
            df = df.append(data)

    df.to_csv('data/results_cd1_1_train.txt', index=False, sep=",")


def select_only_cd2_1_train():
    metadata_train = pd.read_csv('../data/example_result_train.txt', delimiter=",")
    size = metadata_train.shape[0]
    iter = 0

    df_filtered = metadata_train.loc[lambda x: pd.notnull(x['genre_cd2_1'])]
    # for index, data in metadata_train.iterrows():
    #    iter = iter + 1
    #    if iter % 500 == 0:
    #        print(f"{(iter/size)*100}%")
    #    if pd.notnull(data['genre_cd2_1']):
    #        df = df.append(data)
    #
    df_filtered.to_csv('../data/results_cd2_1_rock20_train.txt', index=False, sep=",")


def select_only_20rock_cd1_1_train():
    seed(1)
    metadata_train = pd.read_csv('../data/example_result_train.txt', delimiter=",")
    size = metadata_train.shape[0]
    iter = 0
    metadata_train['randNumCol'] = np.random.randint(0, 20, size=len(metadata_train))
    df_filtered = metadata_train.loc[lambda x: (pd.notnull(x['genre_cd1_1'])) & ((x['genre_cd1_1'] != 'Pop_Rock') | (x['randNumCol'] < 3))]
    print(df_filtered.head(40))
    df_filtered = df_filtered.drop('randNumCol', 1)
    #df_filtered = df_filtered.drop(df_filtered.index[pd.notnull(x['genre_cd1_1']) & ((x['genre_cd1_1'] != 'Pop_Rock') | (np.random.random() < 0.2)) )
    #df_filtered = df_filtered[pd.notnull(df_filtered['genre_cd1_1']) & (np.random.random(5000) <0.2)]
    # for index, data in metadata_train.iterrows():
    #    iter = iter + 1
    #    if iter % 500 == 0:
    #        print(f"{(iter/size)*100}%")
    #    if pd.notnull(data['genre_cd2_1']):
    #        df = df.append(data)
    #
    df_filtered.to_csv('../data/results_cd1_1_rock20_train.txt', index=False, sep=",")


def select_only_20rock_cd2_1_train():
    seed(1)
    metadata_train = pd.read_csv('../data/example_result_train.txt', delimiter=",")
    size = metadata_train.shape[0]
    iter = 0
    metadata_train['randNumCol'] = np.random.randint(0, 20, size=len(metadata_train))
    df_filtered = metadata_train.loc[lambda x: (pd.notnull(x['genre_cd2_1'])) & ((x['genre_cd2_1'] != 'Rock') | (x['randNumCol'] < 5))]
    print(df_filtered.head(40))
    df_filtered = df_filtered.drop('randNumCol', 1)
    #df_filtered = df_filtered.drop(df_filtered.index[pd.notnull(x['genre_cd1_1']) & ((x['genre_cd1_1'] != 'Pop_Rock') | (np.random.random() < 0.2)) )
    #df_filtered = df_filtered[pd.notnull(df_filtered['genre_cd1_1']) & (np.random.random(5000) <0.2)]
    # for index, data in metadata_train.iterrows():
    #    iter = iter + 1
    #    if iter % 500 == 0:
    #        print(f"{(iter/size)*100}%")
    #    if pd.notnull(data['genre_cd2_1']):
    #        df = df.append(data)
    #
    df_filtered.to_csv('../data/results_cd2_1_rock20_train.txt', index=False, sep=",")

def mxm_train_to_cd_count():
    with open("../data/mxm_dataset_train.txt", 'r') as temp_f:
        # get No of columns in each line
        col_count = [len(l.split(",")) for l in temp_f.readlines()]

    ### Generate column names  (names will be 0, 1, 2, ..., maximum columns - 1)
    column_names = [i for i in range(0, max(col_count))]
    # print(column_names)
    ### Read csv
    mxm_train = pd.read_csv('../data/mxm_dataset_train.txt', delimiter=",", names=column_names)
    # print(mxm_train[0])

    with open("../data/msd_tagtraum_cd1.cls", 'r') as temp_f:
        # get No of columns in each line
        col_count = [len(l.split("\t")) for l in temp_f.readlines()]
    column_names = [i for i in range(0, max(col_count))]

    genres_1 = pd.read_csv('../data/msd_tagtraum_cd1.cls', delimiter="\t", names=column_names)
    genres_2 = pd.read_csv('../data/msd_tagtraum_cd2.cls', delimiter="\t", names=column_names)
    # print(genres.head(50))
    counter_pos = 0
    counter_neg = 0
    mxm_train.insert(1, 'genre_cd1_1', '')
    mxm_train.insert(2, 'genre_cd1_2', '')
    mxm_train.insert(3, 'genre_cd2_1', '')
    mxm_train.insert(4, 'genre_cd2_2', '')
    for index, row in mxm_train.iterrows():
        found_cd1 = genres_1[genres_1[0] == row[0]]
        if not found_cd1.empty:
            found_cd1_1_str = found_cd1[1].values[0]
            found_cd1_2_str = found_cd1[2].values[0]
            mxm_train.at[index, 'genre_cd1_1'] = found_cd1_1_str
            mxm_train.at[index, 'genre_cd1_2'] = found_cd1_2_str
            counter_pos = counter_pos + 1

        found_cd2 = genres_2[genres_2[0] == row[0]]
        if not found_cd2.empty:
            found_cd2_1_str = found_cd2[1].values[0]
            found_cd2_2_str = found_cd2[2].values[0]
            counter_pos = counter_pos + 1
            mxm_train.at[index, 'genre_cd2_1'] = found_cd2_1_str
            mxm_train.at[index, 'genre_cd2_2'] = found_cd2_2_str
    mxm_train.to_csv('data/example_result_train.txt', index=False, sep=",")
    print(mxm_train.head(50))
    print(
        f"Found {counter_pos} of {counter_pos + counter_neg} which gives {(counter_pos * 100) / (counter_pos + counter_neg)}%")


def prepare_arrays_of_occurance_test_cd1():
    metadata_test = pd.read_csv('../data/results_cd1_1_test.txt', delimiter=",")
    arr_col_n = 5000
    arr_row_n = metadata_test.shape[0]
    work_array = [[0] * arr_col_n for _ in range(arr_row_n)]
    return work_array


def insert_cd1_test_x_to_array(array):
    metadata_test = pd.read_csv('../data/results_cd1_1_test.txt', delimiter=",")
    for idx, data in metadata_test.iterrows():
        for value in data[6:]:
            if pd.notnull(value):
                idx_to_set, counter = value.split(":")
                array[idx][int(idx_to_set) - 1] = int(counter)
            else:
                break
    return array


def prepare_test_y_cd1():
    metadata_test = pd.read_csv('../data/results_cd1_1_test.txt', delimiter=",")
    return metadata_test['genre_cd1_1']


def prepare_train_y_cd1():
    metadata_train = pd.read_csv('../data/results_cd1_1_rock20_train.txt', delimiter=",")
    return metadata_train['genre_cd1_1']


def prepare_train_y_cd2():
    metadata_train = pd.read_csv('../data/results_cd2_1_rock20_train.txt', delimiter=",")
    return metadata_train['genre_cd2_1']


def prepare_arrays_of_occurance_train_cd1():
    metadata_test = pd.read_csv('../data/results_cd1_1_rock20_train.txt', delimiter=",")
    arr_col_n = 5000
    arr_row_n = metadata_test.shape[0]
    work_array = [[0] * arr_col_n for _ in range(arr_row_n)]
    return work_array


def prepare_arrays_of_occurance_train_cd2():
    metadata_test = pd.read_csv('../data/results_cd2_1_rock20_train.txt', delimiter=",")
    arr_col_n = 5000
    arr_row_n = metadata_test.shape[0]
    work_array = [[0] * arr_col_n for i in range(arr_row_n)]
    return work_array


def insert_cd1_train_x_to_array(array):
    metadata_test = pd.read_csv('../data/results_cd1_1_rock20_train.txt', delimiter=",")
    for idx, data in metadata_test.iterrows():
        for value in data[6:]:
            if pd.notnull(value):
                idx_to_set, counter = value.split(":")
                array[idx][int(idx_to_set) - 1] = int(counter)
            else:
                break
    return array


def insert_cd2_train_x_to_array(array):
    metadata_test = pd.read_csv('../data/results_cd2_1_rock20_train.txt', delimiter=",")
    for idx, data in metadata_test.iterrows():
        for value in data[6:]:
            if pd.notnull(value):
                idx_to_set, counter = value.split(":")
                array[idx][int(idx_to_set) - 1] = int(counter)
            else:
                break
    return array

def check_data():
    arr_train = prepare_arrays_of_occurance_train_cd2()
    train_x = insert_cd2_train_x_to_array(arr_train)
    train_y = prepare_train_y_cd2()
    pd_train_y = pd.array(train_y)
    pd_train_x = pd.array(train_x, dtype=int)

    # pd_test_x.extend(pd_train_x)
    # pd_test_y.extend(pd_train_y)

    x_train, x_test, y_train, y_test = train_test_split(pd_train_x, pd_train_y, test_size=0.2)

    print(Counter(y_train).keys())
    print(Counter(y_train).values())

if __name__ == '__main__':

    select_only_20rock_cd2_1_train()
    check_data()
