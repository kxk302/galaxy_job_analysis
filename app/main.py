from fastapi import FastAPI, APIRouter

import os
import pickle
import sys
# adding scripts folder to the system path
sys.path.append(os.path.join(os.path.dirname(sys.path[0]), 'scripts'))
import regression

import numpy as np
import pandas as pd


app = FastAPI(title='Galaxy Job Memory Prediction API', openapi_url='/openapi.json')
api_router = APIRouter()

models = {}

@app.on_event('startup')
async def startup_event():
    with open('./models/bowtie2.csv_RandomForestRegressor', 'rb') as fp:
      models['bowtie2'] = pickle.load(fp)


@api_router.get('/bowtie2/memory/', status_code=200)
def predict_memory(own_file: int,
                   input_1: int,
                   input_2: int,
                   own_file_extension: str = "fasta",
                   input_1_extension: str = "fastqsanger",
                   input_2_extension: str = "fastqsanger",
                   dbkey: str = "mm10",
                   save_mapping_stats : str = "False",
                   analysis_type_selector: str = "simple",
                   sam_options_selector: str = "none") -> dict:
    tool_name = 'bowtie2'
    loaded_model = models[tool_name]

    params = {
      'own_file': own_file,
      'input_1': input_1,
      'input_2': input_2,
      'own_file_extension': own_file_extension,
      'input_1_extension': input_1_extension,
      'input_2_extension': input_2_extension,
      'dbkey': dbkey,
      'save_mapping_stats': save_mapping_stats,
      'analysis_type_selector': analysis_type_selector,
      'sam_options_selector': sam_options_selector
    }
    print(f'params: {params}')
    df_in = pd.DataFrame(params, index=[0])
    print(f'df_in: {df_in}')
    df = regression.remove_bad_columns(df_in)
    print(f'df: {df}')
    print(f'df.columns.tolist(): {df.columns.tolist()}')

    y_predicted = loaded_model.predict(df)
    # y_predicted is a numpy array. Call tolist() and get the first value from list
    # Call numpy's expm1 to denormalize the value of memory (it was normalized via numpy's log1p)
    y_predicted_denormalized = np.expm1(y_predicted.tolist()[0])

    return {'Tool name': tool_name,
            'Required memory': y_predicted_denormalized}

app.include_router(api_router)


if __name__ == '__main__':
    # Use this for debugging purposes only
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8008, log_level='debug')

