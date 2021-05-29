from math import sqrt
from random import choice
import pandas as pd
import sys


def home_made_mean(feature_values):
    count = 0
    whole_sum = 0
    for value in feature_values:
        count += 1
        whole_sum += value
    return whole_sum / count


def is_numeric(feature_values):
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


description_parameters = {
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


def home_made_describe(features_list):
    list_of_features = []
    for feature in features_list:
        if is_numeric(features_list[feature]):
            list_of_features.append(feature)
            feature_without_na = features_list[feature].dropna()
            sorted_feature = home_made_quicksort(list(feature_without_na))
            feature_count = len(feature_without_na)
            description_parameters["Count"].append(feature_count)
            feature_mean = home_made_mean(feature_without_na)
            description_parameters[""].append(feature_without_na.name)
            description_parameters["Mean"].append(feature_mean)
            description_parameters["Std"].append(home_made_std(feature_mean, feature_count, feature_without_na))
            description_parameters["Min"].append(sorted_feature[0])
            description_parameters["25%"].append(calculate_quantile(25, feature_count, sorted_feature))
            description_parameters["50%"].append(calculate_quantile(50, feature_count, sorted_feature))
            description_parameters["75%"].append(calculate_quantile(75, feature_count, sorted_feature))
            description_parameters["Max"].append(sorted_feature[-1])

    dframe = pd.DataFrame({"count": description_parameters["Count"], "mean": description_parameters["Mean"],
                           "std": description_parameters["Std"], "min": description_parameters["Min"],
                           "25%": description_parameters["25%"], "50%": description_parameters["50%"],
                           "75%": description_parameters["75%"], "max": description_parameters["Max"]},
                          index=list_of_features)
    df_orig = dframe.T
    return df_orig


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
    print(home_made_describe(df))
