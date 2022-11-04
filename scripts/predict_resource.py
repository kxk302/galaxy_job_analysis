import argparse
import json
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

import train_resource_predictor as trp


# input_file:
#   Testing data file name
#
# label_name:
#   Column value to predict in input_file
#
# model_file:
#  Trained model binary to be loaded to make predictions
#
def predict_with_loaded_model(input_file, label_name, model_file, tool_name):
    print(f"input_file: {input_file}")
    df = pd.read_csv(input_file)

    print(f"model_file: {model_file}")
    with open(model_file, "rb") as fp:
        loaded_model = pickle.load(fp)

    df = trp.cleanup_data(df, input_file, label_name)
    if df.shape[0] == 0:
        print(f"No rows in input file {input_file}. Skipping to the next input file")
        return

    X = df.drop(columns=[label_name], axis=1)
    y = df[label_name]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=trp.TEST_SIZE, random_state=trp.RANDOM_STATE
    )

    y_predicted = loaded_model.predict(X_test)
    y_actual = y_test.values

    r2 = r2_score(y_actual, y_predicted)
    print(f"r2: {r2}")
    rmse = mean_squared_error(y_actual, y_predicted, squared=False)
    print(f"rmse: {rmse}")
    mape = mean_absolute_percentage_error(y_actual, y_predicted)
    print(f"mape: {mape}")
    mae = mean_absolute_error(y_actual, y_predicted)
    print(f"mae: {mae}")

    file_name_prefix = "_".join([tool_name, label_name[: label_name.index(".")]])

    actual_vs_predicted_df = pd.DataFrame(
        {"Actual": y_actual, "Predicted": y_predicted}
    )
    actual_vs_predicted_df = actual_vs_predicted_df.astype(int)

    # Ger rid of last few outliers so graph scales are not messed up
    num_of_rows = actual_vs_predicted_df.shape[0]

    actual_vs_predicted_df.to_csv(
        file_name_prefix + "_actual_vs_predicted.tsv",
        sep="\t",
        index=False,
        header=False,
    )
    actual_vs_predicted_df["Actual"].to_csv(
        file_name_prefix + "_actual.tsv", sep="\t", index=False, header=False
    )
    actual_vs_predicted_df["Predicted"].to_csv(
        file_name_prefix + "_predicted.tsv", sep="\t", index=False, header=False
    )
    print(actual_vs_predicted_df.head())
    print(actual_vs_predicted_df.shape)

    plt.scatter(y_actual, y_predicted)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.text(
        21,
        24,
        "R^2: "
        + "{:.2f}".format(r2)
        + ", RMSE: "
        + "{:.2f}".format(rmse)
        + ", MAPE: "
        + "{:.2f}".format(mape),
        fontsize=8,
    )
    plt.show()


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser("Runtime prediction argument parser")
    argument_parser.add_argument("--input_file", "-i", type=str, required=True)
    argument_parser.add_argument("--label_name", "-l", type=str, required=True)
    argument_parser.add_argument("--model_file", "-m", type=str, required=True)
    argument_parser.add_argument("--tool_name", "-t", type=str, required=True)
    args = argument_parser.parse_args()
    predict_with_loaded_model(
        args.input_file, args.label_name, args.model_file, args.tool_name
    )
