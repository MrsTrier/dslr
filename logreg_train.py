import describe
import numpy as np
import pandas as pd
import sys
import plotly.express as px
import  plotly.graph_objects as go


class Model:
    learning_rate = 0.07
    loses = {
        "Hufflepuff": 10,
        "Slytherin": 10,
        "Gryffindor": 10,
        "Ravenclaw": 10
    }
    loses_path = {
        "Hufflepuff": [],
        "Slytherin": [],
        "Gryffindor": [],
        "Ravenclaw": []
    }
    coeffs = {
        "Hufflepuff": [0, 0, 0, 0, 0, 0],
        "Slytherin": [0, 0, 0, 0, 0, 0],
        "Gryffindor": [0, 0, 0, 0, 0, 0],
        "Ravenclaw": [0, 0, 0, 0, 0, 0]
    }
    coeffs_path = {
        "Hufflepuff": {"Constant":[], "Divination": [], "Ancient Runes": [], "Herbology": [], "Charms": [], "Transfiguration": []},
        "Slytherin": {"Constant":[], "Divination": [], "Ancient Runes": [], "Herbology": [], "Charms": [], "Transfiguration": []},
        "Gryffindor": {"Constant":[], "Divination": [], "Ancient Runes": [], "Herbology": [], "Charms": [], "Transfiguration": []},
        "Ravenclaw": {"Constant":[], "Divination": [], "Ancient Runes": [], "Herbology": [], "Charms": [], "Transfiguration": []}
    }


def write_into_file(theta_value_file, values, df):
    try:
        i = 0
        for col in df.iloc[:, 1:7]:
            theta_value_file.write('{}={}\n'.format(col, values[i]))
            i += 1
    except Exception:
        print("Error: something went wrong while writing into file.")


def get_df_for_faculty(faculty):
    df_for_house = df.copy()
    df_for_house['Hogwarts House'] = np.where(df_for_house['Hogwarts House'] == faculty, 1, 0)
    return df_for_house


def update_coeffs(data, model, faculty):
    for index, coeff in enumerate(model.coeffs[faculty]):
        model.coeffs[faculty][index] = coeff - model.learning_rate * ((data['error'] * data.iloc[:, index + 1]).sum() / len(data))
        name = data.iloc[:, index + 1].name
        model.coeffs_path[faculty][name].append(model.coeffs[faculty][index])


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


def stardandaze(course):
    mean = describe.home_made_mean(course)
    return (course - mean) / describe.home_made_std(mean, len(course), course)


def logreg_train():
    model = Model()
    for col in df.iloc[:, 2:7]:
        df[col] = stardandaze(df[col])
    list_of_faculties = df['Hogwarts House'].unique()
    theta_value_file = open('theta_values_file', 'w')
    for indx, faculty in enumerate(list_of_faculties):
        ep = 0
        df_for_faculty = get_df_for_faculty(faculty)
        calculate_error(df_for_faculty, model, faculty)
        enthropy = 0
        while abs(model.loses[faculty] - enthropy) > 0.001:
            model.loses[faculty] = enthropy
            update_coeffs(df_for_faculty, model, faculty)
            calculate_error(df_for_faculty, model, faculty)
            enthropy = - df_for_faculty['enthropy'].sum() / len(df_for_faculty)
            model.loses_path[faculty].append(enthropy)
            ep += 1
        theta_value_file.write('{}\n'.format(faculty))
        write_into_file(theta_value_file, model.coeffs[faculty], df)
    theta_value_file.close()
    return model


def plot_enthropy():
    for indx, faculty in enumerate(df['Hogwarts House'].unique()):
        x = [x + 1 for x in range(len(model.loses_path[faculty]))]
        fig = px.line(y=model.loses_path[faculty], x=x)
        fig.update_layout(title=faculty, xaxis_title="Iteration", yaxis_title="Entropy")
        fig.show()


def plot_coefs():
    for indx, faculty in enumerate(df['Hogwarts House'].unique()):
        fig = go.Figure()
        x = [x + 1 for x in range(len(model.loses_path[faculty]))]
        for col in df.iloc[:, 1:7]:
            fig.add_trace(go.Scatter(x=x, y=model.coeffs_path[faculty][col], mode='lines', name=col))
        fig.update_layout(title=faculty, xaxis_title="Iteration", yaxis_title="Coefficient")
        fig.show()


if __name__ == '__main__':
    df = pd.read_csv(sys.argv[1], index_col='Index')
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = df[["Hogwarts House", "Divination", "Ancient Runes", "Herbology", "Charms", "Transfiguration"]]
    df['Constant'] = 1
    df.insert(df.columns.get_loc("Hogwarts House") + 1, "Constant", df.pop("Constant"))
    model = logreg_train()
    plot_enthropy()
    plot_coefs()
