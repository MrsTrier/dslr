import pandas as pd
import sys


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Error: please give two csv's as parameters.\nFirst - true, and second - prediction")
        exit(0)
    df_truth = pd.read_csv(sys.argv[1], index_col='Index')
    df = pd.read_csv(sys.argv[2], index_col='Index')
    if len(df_truth) != len(df):
        print("Error: dataframes should be of a one length.")
    df['res'] = df['Hogwarts House'] == df_truth["Hogwarts House"]
    print(df['res'].sum()/len(df_truth))
