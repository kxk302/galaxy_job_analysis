## Job Memory Prediction

We can use various Machine Learning (ML) models to predict the memory requirement of various
tools. Training an ML model requires a training dataset. Our training dataset comes from
tracking the tool executions on https://usegalaxy.org -- A comprehensive set of job run
attributes are recorded to CSV files for each tool. The ./examples folder contains 2 such
CSV files for bowtie2 and hisat2 tools. Our goal is to accurately predict the value in the
'memory.max_usage_in_bytes' column, given the values in all the other columns.

The list of tools used in training the ML model is specified in ./config/input_files.csv.
This file contains 2 columns: 'input_file', and 'label'. 'input_file' is the full path to
tool's recorded CSV file. 'label' is the column in the CSV file we are trying to predict (
in case of memory prediction, it's going to be 'memory.max_usage_in_bytes'). We can specify
as many CSV files as we want in the ./config/input_files.csv file -- they will all be used
in training the ML model.

We also want to train different ML models on the same set of tool CSV files, as some models
may perform better than the others. We specify the list of models we want to train in
./config/models.json. The field names in this json file are ML model names (e.g.,
RandomForestRegressor). The field values are attributes for the ML model, e.g., module name
(loaded automatically by the ./scripts/regression.py), class name (instantiated automatically
by the ./scripts/regression.py), and model parameters. We can specify as many parameters as
we want (E.g., for RandomForestRegressor in ./config/models.json, 3 parameters are specified),
and specify as many values as we want for each parameter, and scikit-learn's GridSearchCV (used
by ./scripts/regression.py) will train the model using all the combination of all the parameters
and report the best combination to us. E.g., for RandomForestRegressor, we have specified 3
parameters (n_estimators, max_depth, and max_features), and for each parameter we have specified
2 values. Hence, the model is trained using 2 X 2 X 2 = 8 parametger values and the best model
amongst those 8 would be reported.

### Train ML Model for Memory Prediction

Run the following command to create/activate a virtual environment and install the necessary modules.

```
python3 -m venv venv
. ./venv/bin/activate
pip install -r requirements.txt
```

To train the models specified in ./config/models.json on the training data specified in
./config/input_files.csv, run the following command:

```
python3 ./scripts/regression.py -i ./config/input_files.csv -m ./config/models.json -o ./output_files/output.csv -d ./models
```

The script trains each model specified in ./config/models.json, for all parameter combinations,
and prints the best parameter combination along with the prediction score on training/test dataset.
This information is also saved to an output file (./output_files/output.csv). The model binaries are
saved to ./models folder and can be loaded, e.g. by a REST service to provice endpoints for memory
prediction.

### Serve ML model for Memory Prediction via FastAPI

FastAPI is defined in ./app/main.py. To start the FastAPI, run the following command:

```
uvicorn main:app --reload
```

FastAPI will load the binaries for the best trained model for each tool, located in the ./models folder, in its
startup_event() method. Memory prediction endpoints can be accessed via http://127.0.0.1:8000/docs. For example,
expand /bowtie2/memory/ documentation, specify values for own_file, input_1, and input_2 -- You can use the values
in the first training example in ./examples/bowtie2.csv (own_file: 2102960, input_1: 1493950957, input_2: 0). The
predicted value retunred by the endpoint should be close to the value of memory.max_usage_in_bytes column
(2886856704). The endpoint returned the value of 3661949266 (predicted 3.6GB where 2.8GB was expected).
