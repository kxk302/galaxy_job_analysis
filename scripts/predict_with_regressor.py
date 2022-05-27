import regression

import argparse
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


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
  df_in = pd.read_csv(input_file)

  print(f'model_file: {model_file}')
  with open(model_file, 'rb') as fp:
    loaded_model = pickle.load(fp)

  df = regression.remove_bad_columns(df_in, label_name)

  X = df.drop(columns=[label_name], axis=1)
  y = df[label_name]

  # normalize runtimes
  y = np.log1p(y)

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=regression.TEST_SIZE, random_state=regression.RANDOM_STATE)

  y_predicted = loaded_model.predict(X_test)
  prediction_score = r2_score(y_test, y_predicted)
  print(f'prediction_score: {prediction_score}')


if __name__ == '__main__':
  argument_parser = argparse.ArgumentParser('Runtime prediction argument parser')
  argument_parser.add_argument('--input_file', '-i', type=str, required=True)
  argument_parser.add_argument('--label_name', '-l', type=str, required=True)
  argument_parser.add_argument('--model', '-m', type=str, required=True)
  args = argument_parser.parse_args()
  predict_with_loaded_model(args.input_file, args.label_name, args.model)
