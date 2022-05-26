## Job Runtime Prediction

We can use various Machine Learning (ML) models to predict the runtime of various tools. 
Training an ML model requires a training dataset. Our training dataset comes from tracking 
the tool executions on https://usegalaxy.org -- A comprehensive set of job run attributes 
are recorded to CSV files for each tool. The ./examples folder contains 2 such CSV files 
for BWA-MEM and StringTie tools. Our goal is to accurately predict the value in the 'runtime' 
column, given the values in all the other columns. 

The list of tools used in training the ML model is specified in ./config/input_files.csv.
This file contains 2 columns: 'input_file', and 'label'. 'input_file' is the full path to 
tool's recorded CSV file. 'label' is the column in the CSV file we are trying to predict (
in case of runtime prediction, it's going to be 'runtime').
