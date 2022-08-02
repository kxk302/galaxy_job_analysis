import argparse

import pandas as pd

def generate_input_files(jobs_file, datasets_file, numeric_parameters_file):
  print(f'jobs_file: {jobs_file}')
  print(f'datasets_file: {datasets_file}')
  print(f'numeric_parameters_file: {numeric_parameters_file}')

  jobs_df = pd.read_csv(jobs_file, sep='\t')
  datasets_df = pd.read_csv(datasets_file, sep='\t')
  numeric_parameters_df = pd.read_csv(numeric_parameters_file, sep='\t')

  print(jobs_df.head())
  print(datasets_df.head())
  print(numeric_parameters_df.head())

if __name__ == '__main__':
  argparse = argparse.ArgumentParser()
  argparse.add_argument('--jobs_file', '-j', type=str, required=True)
  argparse.add_argument('--datasets_file', '-d', type=str, required=True)
  argparse.add_argument('--numeric_parameters_file', '-n', type=str, required=True)
  args = argparse.parse_args()
  generate_input_files(args.jobs_file, args.datasets_file, args.numeric_parameters_file) 
