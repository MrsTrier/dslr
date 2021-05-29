import plotly.express as px
import pandas as pd
import sys

def scatter_plot():
    fig = px.scatter(df, x=df['Astronomy'], y=df['Defense Against the Dark Arts'], color="Hogwarts House")
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
    scatter_plot()
