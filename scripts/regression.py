import argparse
import importlib
import json
import os
import pickle

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score     
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

# Percentage of test data
TEST_SIZE = 0.2
# To make random split of training and testing data repeatable
RANDOM_STATE = 13
# Scoring function for regression
SCORING = 'r2'
# Number folds in cross validation for grid search CV
CROSS_VALIDATION_NUM_FOLD=10
# Log level for grid search CV
VERBOSE = 2
# Number of jobs for grid search CV. -1 means use all avilable processors
NUM_JOBS= -1
# To select parameters columns in input file
PARAMETERS = 'parameters.'
# To select file type columns in input file
FILETYPE = '_filetype'
# Ignore this file type when selecting file type columns
IGNORE_FILETYPE = 'chromInfo_filetype'
# List of the begining of bad parameters
BAD_STARTS=['__workflow_invocation_uuid__', 'chromInfo', '__job_resource',
            'reference_source', 'reference_genome', 'rg',
            'readGroup', 'refGenomeSource', 'genomeSource']
# List of the ending of bad parameters
BAD_ENDS = ['id', 'identifier', '__identifier__', 'indeces']
# If more than UNIQUE_CUTOFF of the rows have a unique value, remove the categorical feature
UNIQUE_CUTOFF = 0.5
# If more than NULL_CUTOFF of the rows are null, remove the feature
NULL_CUTOFF = 0.75
# If the number of unique values exceeds NUM_CATEGORIES_CUTOFF, remove the categorical feature
NUM_CATEGORIES_CUTOFF = 100


def remove_bad_columns(df_in, label_name):
  df = df_in.copy()
  num_rows = df.shape[0]

  # Get a list of all user selected parameters
  parameters = [col for col in df.columns.tolist()]

  # Get names of file types and files
  filetypes=[col for col in df.columns.tolist() if (col.endswith(FILETYPE) and col != IGNORE_FILETYPE)]
  files=[filetype[:-len(FILETYPE)] for filetype in filetypes]

  # Remove parameters that start with BAD_STARTS
  for bad_start in BAD_STARTS:
    parameters = [param for param in parameters if not param.startswith(bad_start)]

  # Remove parameters that end with BAD_ENDS
  for bad_end in BAD_ENDS:
    parameters = [param for param in parameters if not param.endswith(bad_end)]

  # Populate list of bad parameters
  bad_parameters = []

  for parameter in parameters:
    series = df[parameter].dropna()

    # Trim string of "". This is necessary to check if the parameter is full of list or dict objects
    if df[parameter].dtype==object and all(type(item)==str and item.startswith('"') and item.endswith('"') for item in series):
      try:
        df[parameter]=df[parameter].str[1:-1].astype(float)
      except:
        df[parameter]=df[parameter].str[1:-1]

    # The following checks are performed on training dataset -- that contains many rows
    # When making prediction, we only have one row, can can skip such set wise filtering
    if num_rows != 1:

      # If more than UNIQUE_CUTOFF of the rows have a unique value, remove the categorical feature
      if df[parameter].dtype==object and len(df[parameter].unique()) >= UNIQUE_CUTOFF*df.shape[0]:
        bad_parameters.append(parameter)

      # If more than NULL_CUTOFF of the rows are null, remove the feature
      if df[parameter].isnull().sum() >= NULL_CUTOFF*df.shape[0]:
        bad_parameters.append(parameter)

      # If the number of categories is greater than NUM_CATEGORIES_CUTOFF remove
      if df[parameter].dtype == object and len(df[parameter].unique()) >= NUM_CATEGORIES_CUTOFF:
        bad_parameters.append(parameter)

    # If the feature is a list remove
    if all(type(item)==str and item.startswith("[") and item.endswith("]") for item in series):
      if all(type(ast.literal_eval(item)) == list for item in series):
        bad_parameters.append(parameter)

    # If the feature is a dict remove
    if all(type(item)==str and item.startswith("{") and item.endswith("}") for item in series):
      if all(type(ast.literal_eval(item)) == dict for item in series):
        bad_parameters.append(parameter)

  for file in files:
    bad_parameters.append("parameters."+file)
    bad_parameters.append("parameters."+file+".values")
    bad_parameters.append("parameters."+file+"|__identifier__")

  for param in set(bad_parameters):
    try:
      parameters.remove(param)
    except:
      pass

  hardware=[label_name]

  keep = parameters + filetypes + files + hardware

  columns = df.columns.tolist()
  for column in columns:
    if not column in keep:
      del df[column]

  return df


# Given a dataframe as input, returns 2 lists,
# one for categorical and one for numerical fetures
def get_categorical_numerical_features(df):
  numerical_features = df.select_dtypes(include='number').columns.tolist()
  categorical_features = df.select_dtypes(exclude='number').columns.tolist()
  return categorical_features, numerical_features


# Defines pipelines for numerical and categorical
# features (to handle missing values, and do
# scaling/encoding). Then defines a ColumnTransformer
# to process dataframe columns based on their type
def get_preprocessor(categorical_features, numerical_features):
  numerical_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='median')),
    ('scale', MinMaxScaler())
  ])

  categorical_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('one-hot', OneHotEncoder(handle_unknown='ignore', sparse=False))
  ])
  
  preprocessor = ColumnTransformer(transformers=[
    ('numerical', numerical_pipeline, numerical_features),
    ('categorical', categorical_pipeline, categorical_features)
  ])

  return preprocessor


# Iterate over each model in models file
#
# models:
#   dictionary containing the contents of models file (model name, module, class name, and parameters)
#
# preprocessor:
#   To preprocess training data
#
# input_file:
#   The models are trained on this training dataset. Goes into output file
#
# label_name:
#  The input_file's column we want to predict
#
# X_train, y_train, X_test, y_test:
#  train and test feature vector and labels
#
# o_file:
#  File pointer to write the output
#
# models_dir:
#   Models directory
#
def process_models(models, preprocessor, input_file, label_name, X_train, y_train, X_test, y_test, o_file, models_dir):
  for model_name in models:
    print(f'model_name: {model_name}')
    module_name = models[model_name]["module_name"]
    class_name = models[model_name]["class_name"]
    parameters = models[model_name]["parameters"]
    print(f'module_name: {module_name}')
    print(f'class_name: {class_name}')
    print(f'parameters: {parameters}')

    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    regressor = class_()

    regressor_pipeline = Pipeline(steps=[
      ('preprocessor', preprocessor),
      ('model', regressor)
    ])

    grid_search_cv = GridSearchCV(regressor_pipeline,
                                  parameters,
                                  scoring=SCORING,
                                  cv=CROSS_VALIDATION_NUM_FOLD,
                                  verbose=VERBOSE,
                                  n_jobs=NUM_JOBS)

    _ = grid_search_cv.fit(X_train, y_train)
    print(f'scorer_: {grid_search_cv.scorer_}')
    print(f'\nRegressor name: {class_name}')
    print(f'input_file: {input_file}, label_name: {label_name}')
    print(f'Best score for {class_name}: {grid_search_cv.best_score_}')
    print(f'Best params for {class_name}: {grid_search_cv.best_params_}')

    best_params_list = [str(k) +'='+ str(v) for k,v in grid_search_cv.best_params_.items()]
    best_params_str = ";".join(best_params_list)
    print(f'best_params_str: {best_params_str}')
    y_predicted = grid_search_cv.predict(X_test)
    prediction_score = r2_score(y_test, y_predicted)
    print(f'prediction_score: {prediction_score}')
    o_file.write(",".join([input_file, model_name, label_name, str(grid_search_cv.best_score_), best_params_str, str(prediction_score)])+'\n')

    # Save model to file
    model_file_name =  os.path.join(models_dir, os.path.basename(input_file) + '_' + model_name)
    with open(model_file_name, 'wb') as fp:
      pickle.dump(grid_search_cv.best_estimator_, fp)


# input_files:
#  Each line in this file specifies the training data file name and the column to be predicted
#
# models_file:
#  Each element in this JSON file specifies the class name of a regressor, the regressor module
#  (to be imported), and the regressor parameters we'd like to do grid search CV on
#
# output_file:
#   This method iterates over each training data file, and then over each regressor model
#   (nested loop) and performs grid search CV. The results (including test data prediction)
#   is written to the output file. The file columns are the following:
#
#   input_file,regressor,label_name,best_score(r2),best_parameters,test_score(r2)
#
def predict(inputs_file, models_file, output_file, models_dir):
  print(f'inputs_file: {inputs_file}')
  df_in = pd.read_csv(inputs_file)
  print(df_in.head())

  print(f'models_file: {models_file}')
  with open(models_file, 'r') as fp:
    models = json.load(fp)
  print(models)

  o_file = open(output_file, 'w')
  o_file.write('input_file,regressor,label_name,best_score(' + SCORING + '),best_parameters,test_score(' + SCORING + ')\n')

  # Iterate over each file in input_files
  for index, row in df_in.iterrows():
    input_file = row['input_file']
    label_name = row['label_name']
    print(f'input_file: {input_file}, label_name: {label_name}')
    df = pd.read_csv(input_file)

    df = remove_bad_columns(df, label_name)

    X = df.drop(columns=[label_name], axis=1)
    y = df[label_name] 
    
    # normalize runtimes
    y = np.log1p(y)

    categorical_features, numerical_features = get_categorical_numerical_features(X)

    num_numerical_features = len(numerical_features)
    print(f'numerical features of the {input_file}')
    print(numerical_features)
    print(f'Number of numerical featuress: {num_numerical_features}')

    num_categorical_features = len(categorical_features)
    print(f'categorical features of the {input_file}')
    print(categorical_features)
    print(f'Number of categorical features: {num_categorical_features}')

    num_features = X.shape[1]
    print(f'Total number of features of the {input_file}: {num_features}')
    if num_numerical_features + num_categorical_features != X.shape[1]:
      raise Exception(f'Number of numerical features ({num_numerical_features}) plus \
              number of categorical features ({num_categorical_features}) \
              does not match the total number of features ({num_features})')
    
    preprocessor = get_preprocessor(categorical_features, numerical_features)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    # Iterate over each model in models file
    process_models(models, preprocessor, input_file, label_name, X_train, y_train, X_test, y_test, o_file, models_dir)

  o_file.close()


if __name__ == '__main__':
  argument_parser = argparse.ArgumentParser('Runtime prediction argument parser')
  argument_parser.add_argument('--input_files', '-i', type=str, required=True)
  argument_parser.add_argument('--models', '-m', type=str, required=True)
  argument_parser.add_argument('--output_file', '-o', type=str, required=True)
  argument_parser.add_argument('--models_dir', '-d', type=str, required=True)
  args = argument_parser.parse_args()
  predict(args.input_files, args.models, args.output_file, args.models_dir)
