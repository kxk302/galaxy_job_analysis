import argparse

import pandas as pd


def get_datasets_fields(df):
  # Call melt on the dataframe so each metric/metric value
  # for a dataset id is in one row as shown below
  #    dataset_id    variable   value
  #      488826   extension     txt
  #      488826   file_size      14
  #      488826  param_name  output
  #      488826        type  output
  #      ...     ...         ...
  df_melt = pd.melt(df,
                    id_vars=['dataset_id'],
                    value_vars=['extension',
                                'file_size',
                                'param_name',
                                'type',
                                'file_size']).sort_values(by=['dataset_id'])
  # Assign a number to each group formed by groupby. This number will be
  # appeneded to metric name to distinguish different metrics of the same
  # type, e.g., input_file_1, input_file_2, input_file_3, etc.
  groupby_keys = list(df_melt.groupby('dataset_id').groups.keys())
  groupby_keys = list(enumerate(groupby_keys))
  df_melt['idx'] = df_melt['dataset_id'].apply(get_groupby_idx, args=[groupby_keys])
  df_melt['variable_idx'] = df_melt['variable'] + '_' + df_melt['idx'].astype(str)
  # variable_idx is all we need. Dropping variable and idx fields
  df_melt.drop(['variable', 'idx'], inplace=True, axis=1)

  # Select only variable_idx and value columns, then transpose it,
  # So, we have variable_idx and value in the first 2 rows
  # variable_idx is read, used as df columns, then deleted (See code below)
  df_t = df_melt[['variable_idx', 'value']].T

  # Call reset_index to get rid of index composed of variable_idx, value
  df_t.reset_index(inplace=True, drop=True)

  # Get the first row in data frame, to be used as column names
  new_columns = df_t.iloc[0,:].tolist()
  df_t.columns = new_columns
  # Delete the first row
  df_t.drop(0, axis=0, inplace=True)
  # Call reset_index after row is deleted
  df_t.reset_index(inplace=True, drop=True)
  return df_t


def get_groupby_idx(dataset_id_in, groupby_keys):
  for (idx, dataset_id) in groupby_keys:
    if dataset_id == dataset_id_in:
      return idx


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

  print('jobs_metrics_df[jobs_metrics_df[\'id\'] == 280155]')
  my_jobs_metrics = jobs_metrics_df[jobs_metrics_df['id'] == 280155]
  print(my_jobs_metrics)
  print(my_jobs_metrics.shape)
  print('\n')
  my_dataset_fields = get_datasets_fields(datasets_df[datasets_df['job_id'] == 280155])
  print('dataset_fields for job_id 280155')
  print(my_dataset_fields)
  print(my_dataset_fields.shape)
  print('\n')

  my_job_metrics_datasets = pd.concat([my_jobs_metrics, my_dataset_fields], axis=1)
  print('jobs_metrics_datasets for job_id 280155')
  print(my_job_metrics_datasets)
  print(my_job_metrics_datasets.shape)


if __name__ == '__main__':
  argparse = argparse.ArgumentParser()
  argparse.add_argument('--jobs_file', '-j', type=str, required=True)
  argparse.add_argument('--datasets_file', '-d', type=str, required=True)
  argparse.add_argument('--numeric_metrics_file', '-n', type=str, required=True)
  args = argparse.parse_args()
  generate_input_files(args.jobs_file, args.datasets_file, args.numeric_metrics_file)