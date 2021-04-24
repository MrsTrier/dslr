import pandas as pd
import sys

def home_made_min(feature_values):
    min_value = feature_values[0]
    for value in feature_values:
        if value < min_value:
            min_value = value
    return min_value

def home_made_max(feature_values):
    max_value = feature_values[0]
    for value in feature_values:
        if value > max_value:
            max_value = value
    return max_value

def isNumeric(feature_values):
    for value in feature_values:
        if type(value) == int or type(value) == float:
            continue
        else:
            return False
    return True

def home_made_sort(feature_values):
    print(feature_values)
    feature_values_copy = feature_values.copy()
    sorted_feature = []
    i = 0

    for e in feature_values_copy:
        # print(feature_values)
        try:
            first = feature_values_copy[i]
        except:
            # sorted_feature.append(feature_values[:1])
            print(feature_values)
            print(feature_values_copy)

            return sorted_feature
        for index, value in enumerate(feature_values_copy):
            if value < first:
                first = value
                index_to_remove = index
        sorted_feature.append(feature_values_copy[i])
        if feature_values_copy is None:
            print(feature_values)
            print(feature_values_copy)
            return sorted_feature
        feature_values_copy.pop(i)
        i += 1
    # print(feature_values)
    # print(feature_values_copy)
    return sorted_feature

def home_made_describe(features_list):
    description_parametrs = {
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
            sorted_feature = home_made_sort(features_list[feature])
            print(sorted_feature)
            # description_parametrs["Min"].append(sorted_feature[0])
            # description_parametrs["Max"].append(sorted_feature[:1])

    # maximums = 'Max '
    # for max in description_parametrs["Max"]:
    #     maximums += '{:>15.3f}'.format(max)
    # print(maximums)
    # mins = 'Min '
    # for min in description_parametrs["Min"]:
    #     mins += '{:>15.3f}'.format(min)
    # print(mins)



    # for feature in features_list:
    #     if isNumeric(features_list[feature]):
    #         description_parametrs["Min"].append(home_made_min(features_list[feature]))
    #         description_parametrs["Max"].append(home_made_max(features_list[feature]))
    #
    # maximums = 'Max '
    # for max in description_parametrs["Max"]:
    #     maximums += '{:>15.3f}'.format(max)
    # print(maximums)
    # mins = 'Min '
    # for min in description_parametrs["Min"]:
    #     mins += '{:>15.3f}'.format(min)
    # print(mins)

if __name__ == '__main__':
    df = pd.read_csv(sys.argv[1])
    # print(df.shape[1])
    home_made_describe(df)
