import math
from math import sqrt
from random import random, choice
import  plotly.graph_objects as go
import plotly.express as px

import numpy as np
import pandas as pd
import sys


def home_made_mean(feature_values):
    count = 0
    whole_sum = 0
    for value in feature_values:
        count += 1
        whole_sum += value
    return whole_sum / count


def isNumeric(feature_values):
    for value in feature_values:
        if type(value) == int or type(value) == float:
            continue
        else:
            return False
    return True


def home_made_std(mean, feature_cont, feature_values):
    whole_sum = 0
    for value in feature_values:
        whole_sum += (value - mean) * (value - mean)
    return sqrt(whole_sum / feature_cont)


def home_made_quicksort(nums):
    if len(nums) <= 1:
        return nums
    else:
        q = choice(nums)
        sorted_nums = []
        max_nums = []
        everage_nums = []
        for n in nums:
            if n < q:
                sorted_nums.append(n)
            elif n > q:
                max_nums.append(n)
            else:
                everage_nums.append(n)
        return home_made_quicksort(sorted_nums) + everage_nums + home_made_quicksort(max_nums)


def calculate_quantile(quantile, count, feature_values):
    indx = quantile * 0.01 * (count + 1)
    return feature_values[round(indx)]


def home_made_describe(features_list):
    description_parametrs = {
        "": [],
        "Count": [],
        "Mean": [],
        "Std": [],
        "Min": [],
        "25%": [],
        "50%": [],
        "75%": [],
        "Max": []
    }
    for feature in features_list:
        if isNumeric(features_list[feature]):
            sorted_feature = home_made_quicksort(features_list[feature].copy())
            feature_count = features_list.shape[0]
            description_parametrs["Count"].append(feature_count)
            feature_mean = home_made_mean(features_list[feature])
            description_parametrs[""].append(features_list[feature].name)
            description_parametrs["Mean"].append(feature_mean)
            description_parametrs["Std"].append(home_made_std(feature_mean, feature_count, features_list[feature]))
            description_parametrs["Min"].append(sorted_feature[0])
            description_parametrs["25%"].append(calculate_quantile(25, feature_count, sorted_feature))
            description_parametrs["50%"].append(calculate_quantile(50, feature_count, sorted_feature))
            description_parametrs["75%"].append(calculate_quantile(75, feature_count, sorted_feature))
            description_parametrs["Max"].append(sorted_feature[-1])
    for key in description_parametrs:
        print_table(description_parametrs, key)


def print_table(description_parametrs, key):
    label = '{} '.format(key)
    for value in description_parametrs[key]:
        if type(value) == str:
            label = '{:<6}{:^25.24s}'.format(label, value)
        else:
            label = '{:<6}{:^25.3f}'.format(label, value)
    print(label)


def plot_histogram():
    house_list = df['Hogwarts House'].unique()
    my_colors = {
        "Hufflepuff": '#dccbf2',
        "Slytherin": '#EDBAB7',
        "Gryffindor": '#CF8867',
        "Ravenclaw": '#6B7D28'
    }
    for course in df.iloc[ : , 5:-1]:
        fig = go.Figure()
        for house in house_list:
            df_for_house = df.loc[df["Hogwarts House"] == house]
            quaniles = []
            feature_count = df_for_house.shape[0]
            df_for_house.reset_index(drop=True, inplace=True)
            sorted_feature = home_made_quicksort(df_for_house[course].copy())
            for x in range(5, 100, 5):
                quaniles.append(calculate_quantile(x, feature_count, sorted_feature))
            fig.add_trace(go.Histogram(x=quaniles, marker_color=my_colors[house]))
        fig.update_layout(barmode='overlay', template="plotly_white")
        fig.update_traces(opacity=0.85)
        fig.show()


def scatter_plot():
    my_colors = {
        "Hufflepuff": '#dccbf2',
        "Slytherin": '#EDBAB7',
        "Gryffindor": '#CF8867',
        "Ravenclaw": '#6B7D28'
    }
    prev_course = df.columns[5]
    for course in df.iloc[ : , 6:]:
        fig = px.scatter(df, x=course, y=prev_course, color="Hogwarts House")
        fig.update_traces(marker=dict(size=7,
                                      line=dict(width=0.2,
                                                color='DarkSlateGrey')),
                          selector=dict(mode='markers'))
        fig.show()
        prev_course = course


def pair_plot():
    fig = px.scatter_matrix(df, dimensions=df.iloc[:, 5:-1], color="Hogwarts House")
    fig.show()
    ### Не брать


# def logreg_predict():
#     try:
#         with open('theta_value_file', 'r') as f:
#             value = f.readlines()
#             index = value[0].index('=')
#             theta0 = value[0][index + 1:]
#             theta1 = value[1][index + 1:]
#     except Exception as e:
#         print("Error: please run ft_linear_regression.py before")
#         print("{}".format(e))
#         exit(0)

def write_into_file(theta_value_file, name, value):
    try:
        theta_value_file.write('{}={}\n'.format(name, value))
    except Exception:
        print("Error: something went wrong while writing into file.")


def get_df_hufflepuff():
    df_for_house = df.loc[df["Hogwarts House"] == "Hufflepuff"]
    df_for_house.reset_index(drop=True, inplace=True)
    df_for_house.loc[df_for_house['Hogwarts House'] == "Hufflepuff", 'Hogwarts House'] = 1
    df_for_house.loc[df_for_house['Hogwarts House'] != "Hufflepuff", 'Hogwarts House'] = 0
    return df_for_house
    # print(df_for_house.head())


def update_coeffs(data, coeffs):
    coeffs[0] = coeffs[0] - learning_rate * pow(data['errors'], 2).sum() / len(data)

def calculate_error(data, coeffs):
    c = 1
    for i in range(len(data)):
        z = 0
        for colmn in data.iloc[:, 1:]:
            z += data.loc[i, colmn] * coeffs[c]
            c += 1
        data[i, 'estimated_y'] = 1/(1 - np.exp(z + coeffs[0]))
        data[i, 'error'] = data[i, 'Hogwarts House'] - data[i, 'estimated_y']


def logreg_train():
    # df = prepare_df(argv)
    i = 0
    learning_rate = 0
    epoch = 2
    hufflepuff_coeffs = [0, 0, 0, 0, 0, 0]
    while i < epoch:
        errors_sum = 0
        df_hufflepuff = get_df_hufflepuff()
        update_coeffs(df_hufflepuff, hufflepuff_coeffs)
        calculate_error(df_hufflepuff, hufflepuff_coeffs)
        for row in df.iterrows():

            print()
            # errors_sum += calculate_error(y, y_head)
        i += 1
    theta_value_file = open('theta_value_file', 'w')
    write_into_file(theta_value_file, "", "")
    theta_value_file.close()


# def prepare_df(argv):
#     df = pd.read_csv(argv, index_col='Index')
#     df.dropna(inplace=True)
#     df.reset_index(drop=True, inplace=True)
#     return df

if __name__ == '__main__':
    df = pd.read_csv(sys.argv[1], index_col='Index')
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    # home_made_describe(df)
    # plot_histogram()
    # scatter_plot()
    # pair_plot()
    df = df[["Hogwarts House", "Divination", "Ancient Runes", "Herbology", "Charms", "Transfiguration"]]
    print(df.head())

    for colmn in df.iloc[:, 1:]:
        df[colmn] = df[colmn] / max(df[colmn])

    # df['Divination'] = df['Divination'] / max(df['Divination'])
    # print(df.head())
    logreg_train()
    # logreg_predict()
