import numpy as np
import pandas as pd
import sys
import logreg_train


class Model:
    coeffs = {
        "Ravenclaw": [0, 0, 0, 0, 0, 0],
        "Slytherin": [0, 0, 0, 0, 0, 0],
        "Gryffindor": [0, 0, 0, 0, 0, 0],
        "Hufflepuff": [0, 0, 0, 0, 0, 0]
    }


def logreg_predict(theta_values_file):
    model = Model()
    list_of_faculties = model.coeffs.keys()
    try:
        with open(theta_values_file, 'r') as f:
            value = f.readlines()
            for indx, faculty in enumerate(list_of_faculties):
                for c in range(6):
                    index = value[indx * 7 + (c + 1)].index('=')
                    model.coeffs[faculty][c] = float(value[indx * 7 + (c + 1)][index + 1:])
    except Exception as e:
        print("Error: please run ft_linear_regression.py before")
        print("{}".format(e))
        exit(0)

    for col in df.iloc[:, 2:7]:
        df[col].fillna(df[col].mean(), inplace=True)
        df[col] = logreg_train.stardandaze(df[col])

    df['Hogwarts House'] = df['Hogwarts House'].fillna('')
    for i in range(df.shape[0]):
        maximum = -1
        for indx, faculty in enumerate(list_of_faculties):
            z = 0
            c = 0
            for col in df.iloc[:, 1:7]:
                z += df.loc[i, col] * model.coeffs[faculty][c]
                c += 1
            if 1/(1 + np.exp(-z)) > maximum:
                maximum = 1/(1 + np.exp(-z))
                df.at[i, 'Hogwarts House'] = faculty
    df.iloc[:, 0:1].to_csv("houses.csv")
    print("Prediction saved into houses.csv")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Error: please give the program dataset_test.csv and a file containing the weights trained by logreg_train.py as a parameters.")
        exit(0)
    df = pd.read_csv(sys.argv[1], index_col='Index')
    df = df[["Hogwarts House", "Divination", "Ancient Runes", "Herbology", "Charms", "Transfiguration"]]
    df['Constant'] = 1
    df.insert(df.columns.get_loc("Hogwarts House") + 1, "Constant", df.pop("Constant"))
    logreg_predict(sys.argv[2])
