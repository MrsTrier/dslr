

import seaborn as sns
from sklearn.linear_model import LogisticRegression

import  plotly.graph_objects as go

import numpy as np
import pandas as pd
import sys

class Model:
    X = []
    Y = []
    learning_rate = 0.01
    thetas_path = []
    loses = {
        "Hufflepuff": 10,
        "Slytherin": 10,
        "Gryffindor": 10,
        "Ravenclaw": 10
    }
    logregs = {
        "Hufflepuff": LogisticRegression(),
        "Slytherin": LogisticRegression(),
        "Gryffindor": LogisticRegression(),
        "Ravenclaw": LogisticRegression()
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




def plot_data_from(model, x):
    # title = '{} as a function of {}'.format(model.Y.name, model.X.name)
    # title = '{}'.format(faculty)

    picture = go.Figure()
    picture.add_trace(go.Scatter(x=x, y=model['Hogwarts House'], mode='markers', name='sample data'))
    # picture.update_layout(xaxis_title=model.X.name, yaxis_title=model.Y.name, title=title)
    picture.add_trace(go.Scatter(x=x, y=model['estimated_y'], mode='markers', name='regression line'))
    picture.show()


def logreg_predict():
    # print(df.head())
    # model = Model()
    list_of_faculties = df['Hogwarts House'].unique()
    try:
        with open('theta_values_file', 'r') as f:
            value = f.readlines()
            for indx, faculty in enumerate(list_of_faculties):
                for c in range(6):
                    index = value[indx * 7 + (c + 1)].index('=')
                    modell.coeffs[faculty][c] = float(value[indx * 7 + (c + 1)][index + 1:])
                print(modell.coeffs[faculty])
    except Exception as e:
        print("Error: please run ft_linear_regression.py before")
        print("{}".format(e))
        exit(0)

    df['estimated_y'] = np.NaN
    # df.estimated_y = df.estimated_y.fillna('')
    df['max_sk'] = np.NaN
    df.max_sk = df.max_sk.fillna('')
    df['max_my'] = np.NaN
    df.max_my = df.max_my.fillna('')
    # print(df.shape[0])
    # print(list_of_faculties)
    s = 0
    for i in range(df.shape[0]):
        maximum = -1
        max_sk = -1
        for indx, faculty in enumerate(list_of_faculties):
            # print(faculty)
            # print(data_to_predict.iloc[:, 1:6])
            y_pred = modell.logregs[faculty].predict(df.iloc[:, 2:7])
            z = 0
            c = 0
            for col in df.iloc[:, 1:7]:
                z += df.loc[i, col] * modell.coeffs[faculty][c]
                c += 1
            if 1/(1 + np.exp(-z)) > maximum:
                maximum = 1/(1 + np.exp(-z))
                # print(faculty)
                df.at[i, 'max_my'] = faculty
            if y_pred[i] > max_sk:
                max_sk = y_pred[i]
                df.at[i, 'max_sk'] = faculty
        if df.at[i, 'max_my'] == df.at[i, 'max_sk']:
            s += 1
    print(s/1251)
        # print("sk {} {}".format(i, df.at[i, 'max_sk']))

    # data_to_predict['estimated_y'] = np.NaN
    # data_to_predict.estimated_y = data_to_predict.estimated_y.fillna('')
    # data_to_predict['max_sk'] = np.NaN
    # data_to_predict.max_sk = data_to_predict.max_sk.fillna('')
    # data_to_predict['max_my'] = np.NaN
    # data_to_predict.max_my = data_to_predict.max_my.fillna('')
    # for i in range(data_to_predict.shape[0]):
    #     maximum = -1
    #     max_sk = -1
    #     for indx, faculty in enumerate(list_of_faculties):
    #         print(faculty)
    #         # print(data_to_predict.iloc[:, 1:6])
    #         y_pred = modell.logregs[faculty].predict(data_to_predict.iloc[:, 1:6])
    #         z = 0
    #         c = 0
    #         for col in data_to_predict.iloc[:, 1:6]:
    #             z += data_to_predict.loc[i, col] * modell.coeffs[faculty][c]
    #             c += 1
    #         if 1/(1 + np.exp(-z)) > maximum:
    #             maximum = 1/(1 + np.exp(-z))
    #             print(faculty)
    #             data_to_predict.at[i, 'max_my'] = faculty
    #         if y_pred[i] > max_sk:
    #             max_sk = y_pred[i]
    #             data_to_predict.at[i, 'max_sk'] = faculty


def write_into_file(theta_value_file, values, df):
    try:
        i = 0
        for col in df.iloc[:, 1:7]:
            theta_value_file.write('{}={}\n'.format(col, values[i]))
            i += 1
    except Exception:
        print("Error: something went wrong while writing into file.")


def get_df_hufflepuff(faculty):
    df_for_house = df.copy()
    # print(df_for_house.head())
    df_for_house['Hogwarts House'] = np.where(df_for_house['Hogwarts House'] == faculty, 1, 0)
    return df_for_house


def update_coeffs(data, model, faculty):
    for index, coeff in enumerate(model.coeffs[faculty]):
        # print("faculty {} {}".format(faculty, coeff))
        # print("faculty {} index {}".format(faculty, data.iloc[:, index + 1]))
        model.coeffs[faculty][index] = coeff - model.learning_rate * ((data['error'] * data.iloc[:, index + 1]).sum() / len(data))
        # print("faculty {} {}".format(faculty, model.coeffs[faculty][index]))
    #     print("-------------")
    # print("=========================")
        # print(model.coeffs[faculty][index])
        # print("faculty {},  index {},  value {}".format(faculty, index, model.coeffs[faculty][index]))


def calculate_error(data, model, faculty):
    data['estimated_y'] = np.NaN
    data['enthropy'] = np.NaN
    for i in range(data.shape[0]):
        z = 0
        c = 0
        for col in data.iloc[:, 1:7]:
            # print(col)
            z += data.loc[i, col] * model.coeffs[faculty][c]
            c += 1
        # print(z)
        data.at[i, 'estimated_y'] = 1/(1 + np.exp(-z))
        if i == 5:
            print(data.at[i, 'estimated_y'])
    data['error'] = data['estimated_y'] - data['Hogwarts House']
    data['enthropy'] = data['Hogwarts House'] * np.log(data['estimated_y']) + (data['Constant'] - data['Hogwarts House']) * np.log(data['Constant'] - data['estimated_y'])

def logreg_train():
    model = Model()
    i = 0
    # print(df["Ancient Runes"])
    df.Divination = (df.Divination - 3.214) / 4.109
    df["Ancient Runes"] = (df["Ancient Runes"] - 496.251) / 106.668
    # print(df["Ancient Runes"])
    df.Herbology = (df.Herbology - 1.189) / 5.221
    df.Charms = (df.Charms + 243.326) / 8.787
    df.Transfiguration = (df.Transfiguration - 1029.863) / 43.965

    list_of_faculties = df['Hogwarts House'].unique()
    theta_value_file = open('theta_values_file', 'w')
    for indx, faculty in enumerate(list_of_faculties):
        ep = 0
        df_for_faculty = get_df_hufflepuff(faculty)
        print(df_for_faculty.head())
        logreg = model.logregs[faculty]
        logreg.fit(df_for_faculty.iloc[:, 2:7], df_for_faculty['Hogwarts House'])
        # # print("intercept {}".format(logreg.intercept_))
        # # print("faculty {} {}".format(faculty, logreg.coef_))
        # y_pred = logreg.predict(df_for_faculty.iloc[:, 2:7])
        # print('log: {}'.format(metrics.accuracy_score(df_for_faculty['Hogwarts House'], y_pred)))

        calculate_error(df_for_faculty, model, faculty)
        # print(df_for_faculty.error)
        enthropy = 0
        # while ep != 1000:
        while abs(model.loses[faculty] - enthropy) > 0.001:
            model.loses[faculty] = enthropy
            update_coeffs(df_for_faculty, model, faculty)
            calculate_error(df_for_faculty, model, faculty)
            # print("{} faculty {}".format(faculty, df_for_faculty.error))
            enthropy = - df_for_faculty['enthropy'].sum() / len(df_for_faculty)
            # print(enthropy)
            ep += 1
        theta_value_file.write('{}\n'.format(faculty))
        write_into_file(theta_value_file, model.coeffs[faculty], df)
        # print(df_for_faculty['estimated_y'])
        # plot_data_from(df_for_faculty, faculty, df_for_faculty.iloc[:, 3])

        # print("{}\n{}".format(faculty, enthropy))
        # print(model.coeffs[faculty])
        # break

    theta_value_file.close()
    return model



if __name__ == '__main__':
    # data_to_predict = pd.read_csv(sys.argv[2], index_col='Index')
    #
    # data_to_predict = data_to_predict[["Divination", "Ancient Runes", "Herbology", "Charms", "Transfiguration"]]
    # data_to_predict['Constant'] = 1
    # data_to_predict.insert(0, "Constant", data_to_predict.pop("Constant"))
    # data_to_predict.dropna(inplace=True)
    # data_to_predict.reset_index(drop=True, inplace=True)
    df = pd.read_csv(sys.argv[1], index_col='Index')
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    df = df[["Hogwarts House", "Divination", "Ancient Runes", "Herbology", "Charms", "Transfiguration"]]
    df['Constant'] = 1
    df.insert(df.columns.get_loc("Hogwarts House") + 1, "Constant", df.pop("Constant"))

    # print(df.head())

    # for colmn in df.iloc[:, 1:]:
    #     df[colmn] = df[colmn] / max(df[colmn])

    modell = logreg_train()
    logreg_predict()

    # picture = go.Figure()
    # picture.add_trace(go.Scatter(x=x, y=data_to_predict['Hogwarts House'], mode='markers', name='sample data'))
    # picture.update_layout(xaxis_title=model.X.name, yaxis_title=model.Y.name, title=title)

    # logreg = LogisticRegression()
    # logreg.fit(df_for_faculty.iloc[:, 2:7], df_for_faculty['Hogwarts House'])
    # print("intercept {}".format(logreg.intercept_))
    # print("faculty {} {}".format(faculty, logreg.coef_))
    # y_pred = logreg.predict(df_for_faculty.iloc[:, 2:7])
    # print('log: {}'.format(metrics.accuracy_score(df_for_faculty['Hogwarts House'], y_pred)))
    # picture.add_trace(go.Scatter(x=data_to_predict.iloc[:, 1], y=data_to_predict['max_sk'], mode='markers', name='regression line'))
    #
    # picture.add_trace(go.Scatter(x=data_to_predict.iloc[:, 1], y=data_to_predict['estimated_y'], mode='markers', name='my estimation'))
    # picture.show()
    #
    # picture = go.Figure()
    # picture.add_trace(go.Scatter(x=data_to_predict.iloc[:, 2], y=data_to_predict['max_sk'], mode='markers', name='regression line'))
    #
    # picture.add_trace(go.Scatter(x=data_to_predict.iloc[:, 2], y=data_to_predict['estimated_y'], mode='markers', name='my estimation'))
    # picture.show()
    #
    # picture = go.Figure()
    # picture.add_trace(go.Scatter(x=data_to_predict.iloc[:, 3], y=data_to_predict['max_sk'], mode='markers', name='regression line'))
    #
    # picture.add_trace(go.Scatter(x=data_to_predict.iloc[:, 3], y=data_to_predict['estimated_y'], mode='markers', name='my estimation'))
    # picture.show()
    # picture = go.Figure()
    # picture.add_trace(go.Scatter(x=data_to_predict.iloc[:, 4], y=data_to_predict['max_sk'], mode='markers', name='regression line'))
    #
    # picture.add_trace(go.Scatter(x=data_to_predict.iloc[:, 4], y=data_to_predict['estimated_y'], mode='markers', name='my estimation'))
    # picture.show()
    # picture = go.Figure()
    # picture.add_trace(go.Scatter(x=data_to_predict.iloc[:, 5], y=data_to_predict['max_sk'], mode='markers', name='regression line'))
    #
    # picture.add_trace(go.Scatter(x=data_to_predict.iloc[:, 5], y=data_to_predict['estimated_y'], mode='markers', name='my estimation'))
    # picture.show()
    # logreg_predict()
