import plotly.express as px
import sys
import pandas as pd


def pair_plot():
    fig = px.scatter_matrix(df, dimensions=df.iloc[:, 5:-1], color="Hogwarts House")
    fig.update_layout(font=dict(family="Courier New, monospace",
                                size=12,
                                color="RebeccaPurple"))
    fig.show()

if __name__ == '__main__':
    if len(sys.argv) == 1:
        csv = input("Enter path to csv: ")
        try:
            df = pd.read_csv(csv, index_col='Index')
        except FileNotFoundError:
            print("Error: file does not exist.")
            exit(0)
        except Exception:
            print("Error: something went wrong. Try another file.")
            exit(0)
    else:
        df = pd.read_csv(sys.argv[1], index_col='Index')
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    pair_plot()
