import argparse

import pandas as pd

def generate_input_files(jobs_file, datasets_file, numeric_metrics_file):
  print('\n')
  print(f'jobs_file: {jobs_file}')
  print(f'datasets_file: {datasets_file}')
  print(f'numeric_metrics_file: {numeric_metrics_file}')
  print('\n')

  jobs_df = pd.read_csv(jobs_file, sep='\t')
  datasets_df = pd.read_csv(datasets_file, sep='\t')
  numeric_metrics_df = pd.read_csv(numeric_metrics_file, sep='\t')

  print('Jobs')
  print(jobs_df.head())
  print('\n')

  print('Datasets')
  print(datasets_df.head())
  print('\n')

  print('Numeric metrics')
  print(numeric_metrics_df.head())
  print('\n')

  # Pivot numeric metrics, so we have one row per job ID
  numeric_metrics_df_piv = numeric_metrics_df[['job_id', 'name', 'value']].pivot('job_id', 'name', 'value')
  print('Pivoted numeric metrics')
  print(numeric_metrics_df_piv.head())
  print('\n')

  # Do inner join between jobs and pivoted numeric metrics
  jobs_metrics_df = pd.merge(jobs_df, numeric_metrics_df_piv, left_on='id', right_on='job_id', how='inner')

  print('Jobs metrics')
  print(jobs_metrics_df.head())
  print(jobs_metrics_df.columns.to_list())
  print('\n')

if __name__ == '__main__':
  argparse = argparse.ArgumentParser()
  argparse.add_argument('--jobs_file', '-j', type=str, required=True)
  argparse.add_argument('--datasets_file', '-d', type=str, required=True)
  argparse.add_argument('--numeric_metrics_file', '-n', type=str, required=True)
  args = argparse.parse_args()
  generate_input_files(args.jobs_file, args.datasets_file, args.numeric_metrics_file)
