import argparse

from train_resource_predictor import CPU_LABEL, get_cpu_utilization

import numpy as np
import pandas as pd

def add_cpu_utilization(input_file, output_file):
    df_in = pd.read_csv(input_file, sep=",", header=0)
    df_in[CPU_LABEL] = get_cpu_utilization(df_in)
    df_in.to_csv(output_file, sep=",", header=True, index=False)

if __name__ == "__main__":
    argparse = argparse.ArgumentParser("Argument parser for " + __name__)
    argparse.add_argument("--input_file", "-i", type=str, required=True)
    argparse.add_argument("--output_file", "-o", type=str, required=True)
    args = argparse.parse_args()
    add_cpu_utilization(args.input_file, args.output_file)

