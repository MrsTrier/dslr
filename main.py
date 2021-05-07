import math
from math import sqrt
from random import random, choice
import  plotly.graph_objects as go
import plotly.express as px

import numpy as np
import pandas as pd
import sys

my_colors = {
    "Hufflepuff": '#dccbf2',
    "Slytherin": '#EDBAB7',
    "Gryffindor": '#CF8867',
    "Ravenclaw": '#6B7D28'
}


class Model:
    X = []
    Y = []
    learning_rate = 0.00001
    thetas_path = []
    loses = {
        "Hufflepuff": 10,
        "Slytherin": 10,
        "Gryffindor": 10,
        "Ravenclaw": 10
    }
    coeffs = {
        "Hufflepuff": [0, 0, 0, 0, 0, 0],
        "Slytherin": [0, 0, 0, 0, 0, 0],
        "Gryffindor": [0, 0, 0, 0, 0, 0],
        "Ravenclaw": [0, 0, 0, 0, 0, 0]
    }
    estimated_Y = []


    def print_coefficients(self):
        print(f'Theta1: { round(self.theta1, 3) }')
        print(f'Theta0: { round(self.theta0, 3) }')

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


def get_df_hufflepuff(faculty):
    df_for_house = df.copy()
    df_for_house['Hogwarts House'] = np.where(df_for_house['Hogwarts House'] == faculty, 1, 0)
    return df_for_house


def update_coeffs(data, model, faculty):
    for index, coeff in enumerate(model.coeffs[faculty]):
        # print(model.coeffs[faculty][index])
        model.coeffs[faculty][index] = coeff - model.learning_rate * ((data['error'] * data.iloc[:, index + 1]).sum() / len(data))
        # print(model.coeffs[faculty][index])
        # print("faculty {},  index {},  value {}".format(faculty, index, model.coeffs[faculty][index]))


def calculate_error(data, model, faculty):
    data['estimated_y'] = np.NaN
    data['enthropy'] = np.NaN
    for i in range(data.shape[0]):
        z = 0
        c = 0
        for col in data.iloc[:, 1:7]:
            z += data.loc[i, col] * model.coeffs[faculty][c]
            c += 1
        data.at[i, 'estimated_y'] = 1/(1 + np.exp(-z))
    data['error'] = data['estimated_y'] - data['Hogwarts House']
    data['enthropy'] = data['Hogwarts House'] * np.log(data['estimated_y']) + (data['Constant'] - data['Hogwarts House']) * np.log(data['Constant'] - data['estimated_y'])


def logreg_train():
    model = Model()
    i = 0
    list_of_faculties = df['Hogwarts House'].unique()
    while list_of_faculties.any():
        for indx, faculty in enumerate(list_of_faculties):
            df_for_faculty = get_df_hufflepuff(faculty)
            # print(df_for_faculty.head())
            calculate_error(df_for_faculty, model, faculty)
            # print(df_for_faculty.head())
            enthropy = - df_for_faculty['enthropy'].sum() / len(df_for_faculty)
            # print(model.loses[faculty])
            if abs(model.loses[faculty] - enthropy) < 0.01:
                list_of_faculties = np.delete(list_of_faculties, indx)
                print("{}\n{}".format(faculty, enthropy))
                print(model.coeffs[faculty])
                break
            model.loses[faculty] = enthropy
            update_coeffs(df_for_faculty, model, faculty)
            i += 1
    print(i)
    theta_value_file = open('theta_value_file', 'w')
    write_into_file(theta_value_file, "", "")
    theta_value_file.close()


if __name__ == '__main__':
    df = pd.read_csv(sys.argv[1], index_col='Index')
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    # home_made_describe(df)
    # plot_histogram()
    # scatter_plot()
    # pair_plot()
    df = df[["Hogwarts House", "Divination", "Ancient Runes", "Herbology", "Charms", "Transfiguration"]]
    df['Constant'] = 1
    df.insert(df.columns.get_loc("Hogwarts House") + 1, "Constant", df.pop("Constant"))

    print(df.head())

    # for colmn in df.iloc[:, 1:]:
    #     df[colmn] = df[colmn] / max(df[colmn])

    logreg_train()
    # logreg_predict()
