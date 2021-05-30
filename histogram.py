import describe
import  plotly.graph_objects as go
import pandas as pd
import sys
from plotly.subplots import make_subplots


my_colors = {
    "Hufflepuff": '#dccbf2',
    "Slytherin": '#EDBAB7',
    "Gryffindor": '#CF8867',
    "Ravenclaw": '#6B7D28'
}


def plot_all_histogram():
    house_list = df['Hogwarts House'].unique()
    for course in df.iloc[:, 5:-1]:
        fig = go.Figure()
        for house in house_list:
            df_for_house = df.loc[df["Hogwarts House"] == house]
            quaniles = []
            feature_count = df_for_house.shape[0]
            df_for_house.reset_index(drop=True, inplace=True)
            sorted_feature = describe.home_made_quicksort(df_for_house[course].copy())
            for x in range(0, 100, 5):
                quaniles.append(describe.calculate_quantile(x, feature_count, sorted_feature))
            fig.add_trace(go.Histogram(x=quaniles, marker_color=my_colors[house], name=house))
        fig.update_layout(title=course, barmode='overlay', template="plotly_white")
        fig.update_traces(opacity=0.85)
        fig.show()


def plot_histogram():
    house_list = df['Hogwarts House'].unique()
    for course in df.iloc[ : , 5:-1]:
        # fig = go.Figure()
        subplot = make_subplots(rows=2, cols=2)

        for index, house in enumerate(house_list):

            df_for_house = df.loc[df["Hogwarts House"] == house]
            quaniles = []
            feature_count = df_for_house.shape[0]
            df_for_house.reset_index(drop=True, inplace=True)
            sorted_feature = describe.home_made_quicksort(df_for_house[course].copy())
            for x in range(5, 100, 2):
                quaniles.append(describe.calculate_quantile(x, feature_count, sorted_feature))
            col = 2 if index==2 or index==3 else 1
            row = index+1 if index < 2 else index-1
            subplot.add_trace(go.Histogram(x=quaniles, marker_color=my_colors[house], name=house), row=row, col=col)
        subplot.update_layout(title=course, barmode='overlay', template="plotly_white")
        subplot.update_traces(opacity=0.85)
        subplot.show()


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
    plot_all_histogram()
