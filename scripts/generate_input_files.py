import argparse
import os

import numpy as np
import pandas as pd

from train_resource_predictor import STATE_COL

"""
def get_datasets_fields(df):
    # Call melt on the dataframe so each metric/metric value
    # for a dataset id is in one row as shown below
    #    dataset_id    variable   value
    #      488826   extension     txt
    #      488826   file_size_bytes      14
    #      488826  param_name  output
    #      488826        type  output
    #      ...     ...         ...
    metrics = ["extension", "file_size_bytes", "param_name", "type"]
    num_of_metrics = len(["extension", "file_size_bytes", "param_name", "type"])

    df_melt = pd.melt(df, id_vars=["dataset_id"], value_vars=metrics).sort_values(
        by=["variable", "dataset_id"]
    )

    num_of_rows = df_melt.shape[0]
    num_of_indexes = num_of_rows / num_of_metrics
    # E.g., if 12 rows in df_melt, and we have 4 metrics, must append 1 to first group
    # of metrics, 2 to second group of metrics, and 3 to third group of metrics
    idx_list = num_of_metrics * np.arange(1, num_of_indexes + 1).tolist()
    df_melt["idx"] = idx_list
    df_melt["variable_idx"] = df_melt["variable"] + "_" + df_melt["idx"].astype(str)
    # variable_idx is all we need. Dropping variable and idx fields
    df_melt.drop(["variable", "idx"], inplace=True, axis=1)

    # Select only variable_idx and value columns, then transpose it,
    # So, we have variable_idx and value in the first 2 rows
    # variable_idx is read, used as df columns, then deleted (See code below)
    df_t = df_melt[["variable_idx", "value"]].T

    # Call reset_index to get rid of index composed of variable_idx, value
    df_t.reset_index(inplace=True, drop=True)

    # Get the first row in data frame, to be used as column names
    new_columns = df_t.iloc[0, :].tolist()
    df_t.columns = new_columns
    # Delete the first row
    df_t.drop(0, axis=0, inplace=True)
    # Call reset_index after row is deleted
    df_t.reset_index(inplace=True, drop=True)
    return df_t
    """


def read_input_files(jobs_file, datasets_file, numeric_metrics_file, separator):
    print("\n")
    print(f"jobs_file: {jobs_file}")
    print(f"datasets_file: {datasets_file}")
    print(f"numeric_metrics_file: {numeric_metrics_file}")
    print("\n")

    jobs_df = pd.read_csv(jobs_file, sep=separator)
    datasets_df = pd.read_csv(datasets_file, sep=separator)
    numeric_metrics_df = pd.read_csv(numeric_metrics_file, sep=separator)

    print("Jobs")
    print(jobs_df.head())
    print("\n")

    print("Datasets")
    print(datasets_df.head())
    print("\n")

    print("Numeric metrics")
    print(numeric_metrics_df.head())
    print("\n")

    return jobs_df, datasets_df, numeric_metrics_df


def generate_input_files(
    jobs_file, datasets_file, numeric_metrics_file, output_dir, separator
):
    jobs_df, datasets_df, numeric_metrics_df = read_input_files(
        jobs_file, datasets_file, numeric_metrics_file, separator
    )

    # We only care about jobs with 'ok' state
    print("jobs_df.shape")
    print(jobs_df.shape)
    print("\n")
    if STATE_COL in jobs_df.columns:
        jobs_df = jobs_df[jobs_df[STATE_COL] == "ok"]
    print('jobs_df.shape with "ok" state')
    print(jobs_df.shape)
    print("\n")

    # Filter output files as in prediction we only care about input files
    print(
        f"Size of dataset_df Prior to filtering oputput files: {datasets_df.shape[0]}"
    )
    datasets_df = datasets_df[datasets_df["type"] != "output"]
    print(
        f"Size of dataset_df After to filtering oputput files: {datasets_df.shape[0]}"
    )

    # Pivot dataset_df so that for each job_id, we have a row of file sizes, one for each input file
    datasets_df_piv = pd.pivot_table(
        datasets_df, index="job_id", columns="param_name", values="file_size_bytes"
    )
    datasets_df_piv.reset_index(inplace=True)

    # Pivot numeric metrics, so we have one row per job ID
    numeric_metrics_df_piv = numeric_metrics_df[
        ["job_id", "metric_name", "metric_value"]
    ].pivot("job_id", "metric_name", "metric_value")
    print("Pivoted numeric metrics")
    print(numeric_metrics_df_piv.head())
    print("\n")

    # Do left join between jobs and pivoted numeric metrics. Left join as some jobs may not have any metrics
    jobs_metrics_df = pd.merge(
        jobs_df, numeric_metrics_df_piv, left_on="job_id", right_on="job_id", how="left"
    )

    print("Jobs metrics")
    print(jobs_metrics_df.head())
    print(jobs_metrics_df.columns.to_list())
    print("\n")

    # Get a list of all tool IDs
    tools_df = (
        jobs_df[["job_id", "tool_id", "tool_version"]]
        .groupby(["tool_id", "tool_version"])
        .count()
        .sort_values(by=["job_id"], ascending=False)
        .reset_index()[["tool_id", "tool_version"]]
    )

    for index, row in tools_df.iterrows():
        a_tool_id = row["tool_id"]
        a_tool_version = row["tool_version"]
        print(f"index: {index}, tool_id: {a_tool_id}, tool_version: {a_tool_version}")
        print("\n")

        # Filter jobs_df based on tool_id and tool_version
        tool_jobs_df = jobs_df[
            (jobs_df["tool_id"] == a_tool_id)
            & (jobs_df["tool_version"] == a_tool_version)
        ]
        # call reset_index so index starts from 0 when iterating over the dataframe
        tool_jobs_df.reset_index(inplace=True)

        a_tool_all_jobs_df = None

        for index_in, row_in in tool_jobs_df.iterrows():
            a_job_id = row_in["job_id"]
            if index_in % 100 == 0:
                print(
                    f"index_in: {index_in}, job_id: {a_job_id}, , tool_id: {a_tool_id}, tool_version: {a_tool_version}"
                )
                print("\n")

            my_dataset_fields = datasets_df_piv[datasets_df_piv["job_id"] == a_job_id]
            my_dataset_fields.reset_index(inplace=True, drop=True)

            my_jobs_metrics_df = jobs_metrics_df[
                (jobs_metrics_df["job_id"] == a_job_id)
                & (jobs_metrics_df["tool_id"] == a_tool_id)
                & (jobs_metrics_df["tool_version"] == a_tool_version)
            ]
            my_jobs_metrics_df.reset_index(drop=True, inplace=True)

            a_tool_a_job_df = pd.concat([my_jobs_metrics_df, my_dataset_fields], axis=1)
            a_tool_a_job_df.reset_index()

            if a_tool_all_jobs_df is None:
                a_tool_all_jobs_df = pd.DataFrame(a_tool_a_job_df)
            else:
                a_tool_all_jobs_df = pd.concat([a_tool_all_jobs_df, a_tool_a_job_df])

        a_tool_all_jobs_df.to_csv(
            output_dir
            + "/"
            + os.path.basename(a_tool_id)
            + "_"
            + a_tool_version
            + ".csv",
            sep=",",
            index=False,
        )


# This script, takes as input the 3 files generated by ./scripts/grt/export.py (script in Galaxy code base)
# and generates a single file with jobs/datasets/metrics for tools all sepecified on one line, to be
# consumed by the learning algorithm (regression.py)
if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--jobs_file", "-j", type=str, required=True)
    argparse.add_argument("--datasets_file", "-d", type=str, required=True)
    argparse.add_argument("--numeric_metrics_file", "-n", type=str, required=True)
    argparse.add_argument("--output_dir", "-o", type=str, required=True)
    argparse.add_argument("--separator", "-s", type=str, required=True)
    args = argparse.parse_args()
    generate_input_files(
        args.jobs_file,
        args.datasets_file,
        args.numeric_metrics_file,
        args.output_dir,
        args.separator,
    )
