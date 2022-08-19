import train_resource_predictor as trp

import argparse
import json
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


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
  print(f'input_file: {input_file}')
  df = pd.read_csv(input_file)

  print(f'model_file: {model_file}')
  with open(model_file, 'rb') as fp:
    loaded_model = pickle.load(fp)

  if label_name == trp.CPU_LABEL:
      df[trp.CPU_LABEL] = trp.get_cpu_utilization(df[ ['cpuacct.usage', 'galaxy_slots', 'runtime_seconds'] ])

  df = trp.cleanup_data(df, input_file, label_name)
  if df.shape[0] == 0:
    print(f'No rows in input file {input_file}. Skipping to the next input file')
    return

  X = df.drop(columns=[label_name], axis=1)
  y = df[label_name]

  # normalize runtimes
  y = np.log1p(y)

  y_predicted = loaded_model.predict(X)
  prediction_score = r2_score(y.values, y_predicted)

  actual_vs_predicted_df = pd.DataFrame({'Actual': y.values, 'Predicted': y_predicted})
  print(actual_vs_predicted_df.head())
  print(actual_vs_predicted_df.shape)
  print(f'prediction_score: {prediction_score}')

  plt.scatter(y.values, y_predicted)
  plt.xlabel('Actual')
  plt.ylabel('Predicted')
  plt.text(21, 24, 'R^2: ' + str(prediction_score), fontsize = 8)
  plt.show()


if __name__ == '__main__':
  argument_parser = argparse.ArgumentParser('Runtime prediction argument parser')
  argument_parser.add_argument('--input_file', '-i', type=str, required=True)
  argument_parser.add_argument('--label_name', '-l', type=str, required=True)
  argument_parser.add_argument('--model', '-m', type=str, required=True)
  args = argument_parser.parse_args()
  predict_with_loaded_model(args.input_file, args.label_name, args.model)
