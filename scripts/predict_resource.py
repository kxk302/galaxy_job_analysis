import argparse
import json
import pickle

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

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
def predict_with_loaded_model(input_file, label_name, model_file):
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

    y_predicted = loaded_model.predict(X)
    y_actual = y.values

    r2 = r2_score(y.values, y_predicted)
    print(f"r2: {r2}")
    rmse = mean_squared_error(y_actual, y_predicted, squared=False)
    print(f"rmse: {rmse}")
    mape = mean_absolute_percentage_error(y_actual, y_predicted)
    print(f"mape: {mape}")
    mae = mean_absolute_error(y_actual, y_predicted)
    print(f"mae: {mae}")

    actual_vs_predicted_df = pd.DataFrame(
        {"Actual": y_actual, "Predicted": y_predicted}
    )
    actual_vs_predicted_df.to_csv("actual_vs_predicted.csv", sep=",")
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
    argument_parser.add_argument("--model", "-m", type=str, required=True)
    args = argument_parser.parse_args()
    predict_with_loaded_model(args.input_file, args.label_name, args.model)
