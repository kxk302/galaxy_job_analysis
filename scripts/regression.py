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
            'readGroup', 'refGenomeSource', 'genomeSource',
            'tool_version', 'state',
            'start_epoch', 'end_epoch',
            'extension_', 'param_name_', 'type_']
# List of the ending of bad parameters
BAD_ENDS = ['id', 'identifier', '__identifier__', 'indeces']
# If more than UNIQUE_CUTOFF of the rows have a unique value, remove the categorical feature
UNIQUE_CUTOFF = 0.5
# If more than NULL_CUTOFF of the rows are null, remove the feature
NULL_CUTOFF = 0.75
# If the number of unique values exceeds NUM_CATEGORIES_CUTOFF, remove the categorical feature
NUM_CATEGORIES_CUTOFF = 100
# Only process rows where state is 'ok'
OK_STATE = 'ok'
# Total features importance cuttoff
FEATURES_IMPORTANCE_CUTOFF = 0.9
# GridSearchCV error_score
ERROR_SCORE = 'raise'


def remove_bad_columns(df_in, label_name = None):
  df = df_in.copy()
  num_rows = df.shape[0]

  # Get a list of all user selected parameters
  parameters = df.columns.tolist()

  # Get names of file types and files
  filetypes=[col for col in parameters if (col.endswith(FILETYPE) and col != IGNORE_FILETYPE)]
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
    bad_parameters.append(file+".values")
    bad_parameters.append(file+"|__identifier__")

  for param in set(bad_parameters):
    try:
      parameters.remove(param)
    except:
      pass

  keep = parameters + filetypes + files + [label_name]

  columns = df.columns.tolist()
  for column in columns:
    if not column in keep:
      del df[column]

  return df


# Drop all memory related columns, besides possibly the label
def remove_memory_columns(df_in, label_name = None):
  df = df_in.copy()

  df_columns = df.columns.tolist()
  df_memory_columns = [column for column in df_columns if 'memory' in column and column != label_name]
  if 'memtotal' in df_columns:
    df_memory_columns.append('memtotal')
  if 'swaptotal' in df_columns:
    df_memory_columns.append('swaptotal')

  df = df.drop(df_memory_columns, axis=1)
  return df


# For numeric columns, ther is no translation
# For categorical columns, with OHE, we go from 1 column to N
# with N being the number of unique values for the categorical column
# This method returns a list, that specifies 1's for numeric columns, and N's
# for categorical columns. This allows us to calculate what a specific column,
# after translation, corresponds to before translation via preprocessor 
def get_column_translation(df):
  column_translation = []
  num_cols_after_translation = 0

  print(f'Number of columns before translation: {df.shape[1]}')
  print(df.columns.tolist())

  for (columnName, columnData) in df.iteritems():
    print('Column Name : ', columnName)
    if df.dtypes[columnName] == 'int64' or df.dtypes[columnName] == 'float64':
      column_translation.append(str(1))
      num_cols_after_translation += 1
    else:
      column_translation.append(str(columnData.nunique()))
      num_cols_after_translation += columnData.nunique()

  return column_translation, num_cols_after_translation


def get_feature_importance(best_estimator_):
  if hasattr(best_estimator_, 'feature_importances_'):
    feature_importances_list = best_estimator_.feature_importances_.tolist()
    feature_importances = list(enumerate(feature_importances_list))
    feature_importances = sorted(feature_importances, reverse=True, key=lambda x: x[1])

    fi_df = pd.DataFrame({'feature_index': [k for k,v in feature_importances],
                            'feature_importance': [v for k,v in feature_importances]})
    fi_df['feature_importance_cumulative_sum'] = fi_df['feature_importance'].cumsum()
    print('feature importance df')
    print(fi_df)
    fi_df = fi_df[ fi_df['feature_importance_cumulative_sum'] <= FEATURES_IMPORTANCE_CUTOFF]
    print('feature importance df AFTER cutoff applied')
    print(fi_df)
    fi_df['index_importance'] = fi_df['feature_index'].astype(str) + ":" + fi_df['feature_importance'].astype(str)
    print('feature importance df AFTER composite column')
    print(fi_df)

    feature_importances = fi_df['index_importance'].tolist()
    feature_importances = ";".join(feature_importances)
    print(f'feature_importances: {feature_importances}')
    return feature_importances
  else:
    print(f'{best_estimator_} does not have feature_importances_ property')
    return ""


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
                                  n_jobs=NUM_JOBS,
                                  error_score=ERROR_SCORE)

    _ = grid_search_cv.fit(X_train, y_train)
    print(f'scorer_: {grid_search_cv.scorer_}')
    print(f'\nRegressor name: {class_name}')
    print(f'input_file: {input_file}, label_name: {label_name}')
    print(f'Best score for {class_name}: {grid_search_cv.best_score_}')
    print(f'Best params for {class_name}: {grid_search_cv.best_params_}')

    # Calculate feature importance
    feature_importance = get_feature_importance(grid_search_cv.best_estimator_[1])

    # Get number of unique values for each categorical column
    # So, you can translate the feature vector, before and after
    # applying the preprocessor (e.g., when going from 44 columns
    # before preprocessor to 64 columns after preprocessor)
    # For numerical columns, just state 1
    # For categorical columns, state the number of unique values
    column_translation, num_cols_after_translation = get_column_translation(X_train)
    print('column_translation')
    print(column_translation)

    best_params_list = [str(k) +'='+ str(v) for k,v in grid_search_cv.best_params_.items()]
    best_params_str = ";".join(best_params_list)
    print(f'best_params_str: {best_params_str}')
    y_predicted = grid_search_cv.predict(X_test)
    prediction_score = r2_score(y_test, y_predicted)
    print(f'prediction_score: {prediction_score}')
    o_file.write(",".join([input_file,
                           model_name,
                           label_name,
                           str(grid_search_cv.scorer_),
                           str(grid_search_cv.best_score_),
                           best_params_str,
                           str(prediction_score),
                           str(len(grid_search_cv.feature_names_in_.tolist())),
                           ":".join(grid_search_cv.feature_names_in_.tolist()),
                           feature_importance,
                           str(num_cols_after_translation),
                           ":".join(column_translation)])+'\n')

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
  df_in = pd.read_csv(inputs_file, comment='#')
  print(df_in.head())

  print(f'models_file: {models_file}')
  with open(models_file, 'r') as fp:
    models = json.load(fp)
  print(models)

  o_file = open(output_file, 'w')
  o_file.write('input_file,regressor,label_name,scorer_function,best_score,best_parameters,test_score,num_features,feature_names,feature_importance,num_cols_after_translation,column_translation\n')

  # Iterate over each file in input_files
  for index, row in df_in.iterrows():
    input_file = row['input_file']
    label_name = row['label_name']
    print(f'input_file: {input_file}, label_name: {label_name}')
    df = pd.read_csv(input_file)

    # Only process rows where state is 'ok'
    df = df[ df['state'] == OK_STATE ]

    # Only keep rows that the label column is not null
    df = df[ df[label_name].notnull() ]

    # Remove bad columns
    df = remove_bad_columns(df, label_name)

    # Remove memory columns (Only when we are predicting memory)
    df = remove_memory_columns(df, label_name)

    if df.shape[0] == 0:
      print(f'No rows in input file {input_file}. Skipping to the next input file')
      continue

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
  argument_parser = argparse.ArgumentParser('Resource prediction argument parser')
  argument_parser.add_argument('--input_files', '-i', type=str, required=True)
  argument_parser.add_argument('--models', '-m', type=str, required=True)
  argument_parser.add_argument('--output_file', '-o', type=str, required=True)
  argument_parser.add_argument('--models_dir', '-d', type=str, required=True)
  args = argument_parser.parse_args()
  predict(args.input_files, args.models, args.output_file, args.models_dir)
