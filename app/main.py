import os
import pickle
import sys

import numpy as np
import pandas as pd
from fastapi import APIRouter, FastAPI

app = FastAPI(title="Galaxy Tool Resource Prediction API", openapi_url="/openapi.json")
api_router = APIRouter()

models = {}


@app.on_event("startup")
async def startup_event():
    with open(
        "../models/bowtie2/mem/mem_2.4.2+galaxy0_2.4.2+galaxy0.tsv_BaggingRegressor",
        "rb",
    ) as fp:
        models["bowtie2-mem"] = pickle.load(fp)
    with open(
        "../models/bowtie2/cpu/cpu_2.4.2+galaxy0_2.4.2+galaxy0.tsv_BaggingRegressor",
        "rb",
    ) as fp:
        models["bowtie2-cpu"] = pickle.load(fp)
    with open(
        "../models/bwa_mem/mem/mem_0.7.17.2_0.7.17.2.tsv_BaggingRegressor",
        "rb",
    ) as fp:
        models["bwa_mem-mem"] = pickle.load(fp)
    with open(
        "../models/bwa_mem/cpu/cpu_0.7.17.2_0.7.17.2.tsv_BaggingRegressor",
        "rb",
    ) as fp:
        models["bwa_mem-cpu"] = pickle.load(fp)
    with open(
        "../models/fastqc/mem/mem_0.73+galaxy0_0.73+galaxy0.tsv_BaggingRegressor",
        "rb",
    ) as fp:
        models["fastqc-mem"] = pickle.load(fp)
    with open(
        "../models/fastqc/cpu/cpu_0.73+galaxy0_0.73+galaxy0.tsv_GradientBoostingRegressor",
        "rb",
    ) as fp:
        models["fastqc-cpu"] = pickle.load(fp)
    with open(
        "../models/minimap2/mem/mem_2.24+galaxy0_2.24+galaxy0.tsv_BaggingRegressor",
        "rb",
    ) as fp:
        models["minimap2-mem"] = pickle.load(fp)
    with open(
        "../models/minimap2/cpu/cpu_2.24+galaxy0_2.24+galaxy0.tsv_BaggingRegressor",
        "rb",
    ) as fp:
        models["minimap2-cpu"] = pickle.load(fp)
    # with open(
    #     "../models/hisat2/mem/mem_2.2.1+galaxy0_2.2.1+galaxy0.tsv_RandomForestRegressor",
    #     "rb",
    # ) as fp:
    #     models["hisat2-mem"] = pickle.load(fp)
    # with open(
    #     "../models/hisat2/cpu/cpu_2.2.1+galaxy0_2.2.1+galaxy0.tsv_GradientBoostingRegressor",
    #     "rb",
    # ) as fp:
    #     models["hisat2-cpu"] = pickle.load(fp)
    # with open(
    #     "../models/rna_star/mem/mem_2.7.8a+galaxy0_2.7.8a+galaxy0.tsv_GradientBoostingRegressor",
    #     "rb",
    # ) as fp:
    #     models["rna_star-mem"] = pickle.load(fp)
    # with open(
    #     "../models/rna_star/cpu/cpu_2.7.8a+galaxy0_2.7.8a+galaxy0.tsv_GradientBoostingRegressor",
    #     "rb",
    # ) as fp:
    #     models["rna_star-cpu"] = pickle.load(fp)
    # with open(
    #     "../models/trimmomatic/mem/mem_0.38.0_0.38.0.tsv_GradientBoostingRegressor",
    #     "rb",
    # ) as fp:
    #     models["trimmomatic-mem"] = pickle.load(fp)
    # with open(
    #     "../models/trimmomatic/cpu/cpu_0.38.0_0.38.0.tsv_GradientBoostingRegressor",
    #     "rb",
    # ) as fp:
    #     models["trimmomatic-cpu"] = pickle.load(fp)
    # with open(
    #     "../models/multiqc/mem/",
    #     "rb",
    # ) as fp:
    #     models["multiqc-mem"] = pickle.load(fp)
    # with open(
    #     "../models/multiqc/cpu/",
    #     "rb",
    # ) as fp:
    #     models["multiqc-cpu"] = pickle.load(fp)
    # with open(
    #     "../models/deseq2/mem/",
    #     "rb",
    # ) as fp:
    #     models["deseq2-mem"] = pickle.load(fp)
    # with open(
    #     "../models/deseq2/cpu/",
    #     "rb",
    # ) as fp:
    #     models["deseq2-cpu"] = pickle.load(fp)
    with open(
        "../models/featurecounts/mem/mem_2.0.1+galaxy2_2.0.1+galaxy2.tsv_GradientBoostingRegressor",
        "rb",
    ) as fp:
        models["featurecounts-mem"] = pickle.load(fp)
    with open(
        "../models/featurecounts/cpu/cpu_2.0.1+galaxy2_2.0.1+galaxy2.tsv_BaggingRegressor",
        "rb",
    ) as fp:
        models["featurecounts-cpu"] = pickle.load(fp)
    # with open(
    #     "../models/freebayes/mem/mem_1.3.1_1.3.1.tsv_GradientBoostingRegressor",
    #     "rb",
    # ) as fp:
    #     models["freebayes-mem"] = pickle.load(fp)
    # with open(
    #     "../models/freebayes/cpu/cpu_1.3.1_1.3.1.tsv_GradientBoostingRegressor",
    #     "rb",
    # ) as fp:
    #     models["freebayes-cpu"] = pickle.load(fp)


# Takes in a number in bytes, and return a string
# representation in Kilo, Mega, or Giga bytes
def to_kilo_mega_giga_bytes(mem_bytes):
    if mem_bytes < 2**10:
        return str(mem_bytes)
    elif mem_bytes >= 2**10 and mem_bytes < 2**20:
        return str(round(mem_bytes / 2**10, 2)) + "KB"
    elif mem_bytes >= 2**20 and mem_bytes < 2**30:
        return str(round(mem_bytes / 2**20, 2)) + "MB"
    elif mem_bytes >= 2**30:
        return str(round(mem_bytes / 2**30, 2)) + "GB"


###########
## bowtie2
###########
@api_router.get("/bowtie2/memory/", status_code=200)
def predict_bowtie2_memory(
    fastqsanger_file_size_bytes_1: int = 0,
    fastqsanger_file_size_bytes_2: int = 0,
    fastqsanger_gz_file_size_bytes_1: int = 0,
    fastqsanger_gz_file_size_bytes_2: int = 0,
    fasta_file_size_bytes: int = 0,
    format_output: bool = False,
) -> dict:
    tool_name = "bowtie2-mem"
    loaded_model = models[tool_name]

    params = {
        "input_1": float(fastqsanger_file_size_bytes_1),
        "input_11": float(fastqsanger_gz_file_size_bytes_1),
        "input_12": float(fastqsanger_gz_file_size_bytes_2),
        "input_2": float(fastqsanger_file_size_bytes_2),
        "own_file": float(fasta_file_size_bytes),
    }

    # The default value for input parameters is 0
    # The model expects nan for 0 values

    for key in params:
        if params[key] == 0:
            params[key] = np.nan
    print(f"params: {params}")
    df = pd.DataFrame(params, index=[0])
    print(f"df: {df}")

    y_predicted = loaded_model.predict(df)
    # y_predicted is a numpy array. Call tolist() and get the first value from list
    y_predicted_denormalized = y_predicted.tolist()[0]

    if format_output:
        y_predicted_denormalized = to_kilo_mega_giga_bytes(y_predicted_denormalized)

    return {
        "Tool name": tool_name,
        "Required memory": y_predicted_denormalized,
    }


@api_router.get("/bowtie2/cpu/", status_code=200)
def predict_bowtie2_cpu(
    fastqsanger_file_size_bytes_1: int = 0,
    fastqsanger_file_size_bytes_2: int = 0,
    fastqsanger_gz_file_size_bytes_1: int = 0,
    fastqsanger_gz_file_size_bytes_2: int = 0,
    fasta_file_size_bytes: int = 0,
) -> dict:

    result = predict_bowtie2_memory(
        fastqsanger_file_size_bytes_1,
        fastqsanger_file_size_bytes_2,
        fastqsanger_gz_file_size_bytes_1,
        fastqsanger_gz_file_size_bytes_2,
        fasta_file_size_bytes,
    )

    tool_name = "bowtie2-cpu"
    loaded_model = models[tool_name]

    params = {
        "input_1": float(fastqsanger_file_size_bytes_1),
        "input_2": float(fastqsanger_file_size_bytes_2),
        "input_11": float(fastqsanger_gz_file_size_bytes_1),
        "input_12": float(fastqsanger_gz_file_size_bytes_2),
        "own_file": float(fasta_file_size_bytes),
        "memory.max_usage_in_bytes": float(result["Required memory"]),
    }

    # The default value for input parameters is 0
    # The model expects nan for 0 values
    for key in params:
        if params[key] == 0:
            params[key] = np.nan

    print(f"params: {params}")
    df = pd.DataFrame(params, index=[0])
    print(f"df: {df}")

    y_predicted = loaded_model.predict(df)
    # y_predicted is a numpy array. Call tolist() and get the first value from list
    y_predicted_denormalized = y_predicted.tolist()[0]

    return {"Tool name": tool_name, "Required CPU": y_predicted_denormalized}


###########
## bwa_mem
###########
@api_router.get("/bwa_mem/memory/", status_code=200)
def predict_bwa_mem_memory(
    fastq_input1: int = 0,
    fastq_input11: int = 0,
    fastq_input12: int = 0,
    fastq_input2: int = 0,
    ref_file: int = 0,
    format_output: bool = False,
) -> dict:
    tool_name = "bwa_mem-mem"
    loaded_model = models[tool_name]

    params = {
        "fastq_input1": float(fastq_input1),
        "fastq_input11": float(fastq_input11),
        "fastq_input12": float(fastq_input12),
        "fastq_input2": float(fastq_input2),
        "ref_file": float(ref_file),
    }

    # The default value for input parameters is 0
    # The model expects nan for 0 values
    for key in params:
        if params[key] == 0:
            params[key] = np.nan

    print(f"params: {params}")
    df = pd.DataFrame(params, index=[0])
    print(f"df: {df}")

    y_predicted = loaded_model.predict(df)
    # y_predicted is a numpy array. Call tolist() and get the first value from list
    y_predicted_denormalized = y_predicted.tolist()[0]

    if format_output:
        y_predicted_denormalized = to_kilo_mega_giga_bytes(y_predicted_denormalized)

    return {"Tool name": tool_name, "Required memory": y_predicted_denormalized}


@api_router.get("/bwa_mem/cpu/", status_code=200)
def predict_bwa_mem_cpu(
    fastq_input1: int = 0,
    fastq_input11: int = 0,
    fastq_input12: int = 0,
    fastq_input2: int = 0,
    ref_file: int = 0,
) -> dict:

    result = predict_bwa_mem_memory(
        fastq_input1,
        fastq_input11,
        fastq_input12,
        fastq_input2,
        ref_file,
    )

    tool_name = "bwa_mem-cpu"
    loaded_model = models[tool_name]

    params = {
        "fastq_input1": float(fastq_input1),
        "fastq_input11": float(fastq_input11),
        "fastq_input12": float(fastq_input12),
        "fastq_input2": float(fastq_input2),
        "ref_file": float(ref_file),
        "memory.max_usage_in_bytes": float(result["Required memory"]),
    }

    # The default value for input parameters is 0
    # The model expects nan for 0 values
    for key in params:
        if params[key] == 0:
            params[key] = np.nan

    print(f"params: {params}")
    df = pd.DataFrame(params, index=[0])
    print(f"df: {df}")

    y_predicted = loaded_model.predict(df)
    # y_predicted is a numpy array. Call tolist() and get the first value from list
    y_predicted_denormalized = y_predicted.tolist()[0]

    return {"Tool name": tool_name, "Required CPU": y_predicted_denormalized}


###########
## fastqc
###########
@api_router.get("/fastqc/memory/", status_code=200)
def predict_fastqc_memory(
    adapters: int = 0,
    contaminants: int = 0,
    input_file: int = 0,
    limits: int = 0,
    format_output: bool = False,
) -> dict:
    tool_name = "fastqc-mem"
    loaded_model = models[tool_name]

    params = {
        "adapters": float(adapters),
        "contaminants": float(contaminants),
        "input_file": float(input_file),
        "limits": float(limits),
    }

    # The default value for input parameters is 0
    # The model expects nan for 0 values
    for key in params:
        if params[key] == 0:
            params[key] = np.nan

    print(f"params: {params}")
    df = pd.DataFrame(params, index=[0])
    print(f"df: {df}")

    y_predicted = loaded_model.predict(df)
    # y_predicted is a numpy array. Call tolist() and get the first value from list
    y_predicted_denormalized = y_predicted.tolist()[0]

    if format_output:
        y_predicted_denormalized = to_kilo_mega_giga_bytes(y_predicted_denormalized)

    return {"Tool name": tool_name, "Required memory": y_predicted_denormalized}


@api_router.get("/fastqc/cpu/", status_code=200)
def predict_fastqc_cpu(
    adapters: int = 0, contaminants: int = 0, input_file: int = 0, limits: int = 0
) -> dict:

    result = predict_fastqc_memory(adapters, contaminants, input_file, limits)

    tool_name = "fastqc-cpu"
    loaded_model = models[tool_name]

    params = {
        "adapters": float(adapters),
        "contaminants": float(contaminants),
        "input_file": float(input_file),
        "limits": float(limits),
        "memory.max_usage_in_bytes": float(result["Required memory"]),
    }

    # The default value for input parameters is 0
    # The model expects nan for 0 values
    for key in params:
        if params[key] == 0:
            params[key] = np.nan

    print(f"params: {params}")
    df = pd.DataFrame(params, index=[0])
    print(f"df: {df}")

    y_predicted = loaded_model.predict(df)
    # y_predicted is a numpy array. Call tolist() and get the first value from list
    y_predicted_denormalized = y_predicted.tolist()[0]

    return {"Tool name": tool_name, "Required CPU": y_predicted_denormalized}


############
## minimap2
############
@api_router.get("/minimap2/memory/", status_code=200)
def predict_minimap2_memory(
    fastq_input1: int = 0,
    fastq_input11: int = 0,
    fastq_input12: int = 0,
    fastq_input2: int = 0,
    ref_file: int = 0,
    format_output: bool = False,
) -> dict:
    tool_name = "minimap2-mem"
    loaded_model = models[tool_name]

    params = {
        "fastq_input1": float(fastq_input1),
        "fastq_input11": float(fastq_input11),
        "fastq_input12": float(fastq_input12),
        "fastq_input2": float(fastq_input2),
        "ref_file": float(ref_file),
        "alignment_options|splicing|junc_bed": float(
            0
        ),  # redundant input? Investigate why it is needed.
    }

    # The default value for input parameters is 0
    # The model expects nan for 0 values
    for key in params:
        if params[key] == 0:
            params[key] = np.nan

    print(f"params: {params}")
    df = pd.DataFrame(params, index=[0])
    print(f"df: {df}")

    y_predicted = loaded_model.predict(df)
    # y_predicted is a numpy array. Call tolist() and get the first value from list
    y_predicted_denormalized = y_predicted.tolist()[0]

    if format_output:
        y_predicted_denormalized = to_kilo_mega_giga_bytes(y_predicted_denormalized)

    return {"Tool name": tool_name, "Required memory": y_predicted_denormalized}


@api_router.get("/minimap2/cpu/", status_code=200)
def predict_minimap2_cpu(
    fastq_input1: int = 0,
    fastq_input11: int = 0,
    fastq_input12: int = 0,
    fastq_input2: int = 0,
    ref_file: int = 0,
) -> dict:

    result = predict_minimap2_memory(
        fastq_input1, fastq_input11, fastq_input12, fastq_input2, ref_file
    )

    tool_name = "minimap2-cpu"
    loaded_model = models[tool_name]

    params = {
        "fastq_input1": float(fastq_input1),
        "fastq_input11": float(fastq_input11),
        "fastq_input12": float(fastq_input12),
        "fastq_input2": float(fastq_input2),
        "ref_file": float(ref_file),
        "memory.max_usage_in_bytes": result["Required memory"],
        "alignment_options|splicing|junc_bed": float(
            0
        ),  # redundant input? Investigate why it is needed.
    }

    # The default value for input parameters is 0
    # The model expects nan for 0 values
    for key in params:
        if params[key] == 0:
            params[key] = np.nan

    print(f"params: {params}")
    df = pd.DataFrame(params, index=[0])
    print(f"df: {df}")

    y_predicted = loaded_model.predict(df)
    # y_predicted is a numpy array. Call tolist() and get the first value from list
    y_predicted_denormalized = y_predicted.tolist()[0]

    return {"Tool name": tool_name, "Required CPU": y_predicted_denormalized}


###########
## hisat2
###########
"""
@api_router.get("/hisat2/memory/", status_code=200)
def predict_hisat2_memory(
    file_size_bytes_1: int, file_size_bytes_2: int, format_output: bool = False
) -> dict:
    tool_name = "hisat2-mem"
    loaded_model = models[tool_name]

    params = {
        "file_size_bytes_1.0": file_size_bytes_1,
        "file_size_bytes_2.0": file_size_bytes_2,
    }
    print(f"params: {params}")
    df = pd.DataFrame(params, index=[0])
    print(f"df: {df}")

    y_predicted = loaded_model.predict(df)
    # y_predicted is a numpy array. Call tolist() and get the first value from list
    y_predicted_denormalized = y_predicted.tolist()[0]

    if format_output:
        y_predicted_denormalized = to_kilo_mega_giga_bytes(y_predicted_denormalized)

    return {"Tool name": tool_name, "Required memory": y_predicted_denormalized}


@api_router.get("/hisat2/cpu/", status_code=200)
def predict_hisat2_cpu(file_size_bytes_1: int, file_size_bytes_2: int) -> dict:

    result = predict_hisat2_memory(file_size_bytes_1, file_size_bytes_2)

    tool_name = "hisat2-cpu"
    loaded_model = models[tool_name]

    params = {
        "file_size_bytes_1.0": file_size_bytes_1,
        "file_size_bytes_2.0": file_size_bytes_2,
        "memory.max_usage_in_bytes": result["Required memory"],
    }
    print(f"params: {params}")
    df = pd.DataFrame(params, index=[0])
    print(f"df: {df}")

    y_predicted = loaded_model.predict(df)
    # y_predicted is a numpy array. Call tolist() and get the first value from list
    y_predicted_denormalized = y_predicted.tolist()[0]

    return {"Tool name": tool_name, "Required CPU": y_predicted_denormalized}
"""


############
## rna_star
############
"""
@api_router.get("/rna_star/memory/", status_code=200)
def predict_rna_star_memory(
    file_size_bytes_1: int,
    file_size_bytes_2: int,
    file_size_bytes_3: int,
    format_output: bool = False,
) -> dict:
    tool_name = "rna_star-mem"
    loaded_model = models[tool_name]

    params = {
        "file_size_bytes_1.0": file_size_bytes_1,
        "file_size_bytes_2.0": file_size_bytes_2,
        "file_size_bytes_3.0": file_size_bytes_3,
    }
    print(f"params: {params}")
    df = pd.DataFrame(params, index=[0])
    print(f"df: {df}")

    y_predicted = loaded_model.predict(df)
    # y_predicted is a numpy array. Call tolist() and get the first value from list
    y_predicted_denormalized = y_predicted.tolist()[0]

    if format_output:
        y_predicted_denormalized = to_kilo_mega_giga_bytes(y_predicted_denormalized)

    return {"Tool name": tool_name, "Required memory": y_predicted_denormalized}


@api_router.get("/rna_star/cpu/", status_code=200)
def predict_rna_star_cpu(
    file_size_bytes_1: int, file_size_bytes_2: int, file_size_bytes_3: int
) -> dict:

    result = predict_rna_star_memory(
        file_size_bytes_1, file_size_bytes_2, file_size_bytes_3
    )

    tool_name = "rna_star-cpu"
    loaded_model = models[tool_name]

    params = {
        "file_size_bytes_1.0": file_size_bytes_1,
        "file_size_bytes_2.0": file_size_bytes_2,
        "file_size_bytes_3.0": file_size_bytes_3,
        "memory.max_usage_in_bytes": result["Required memory"],
    }
    print(f"params: {params}")
    df = pd.DataFrame(params, index=[0])
    print(f"df: {df}")

    y_predicted = loaded_model.predict(df)
    # y_predicted is a numpy array. Call tolist() and get the first value from list
    y_predicted_denormalized = y_predicted.tolist()[0]

    return {"Tool name": tool_name, "Required CPU": y_predicted_denormalized}
"""


###############
## trimmomatic
###############
"""
@api_router.get("/trimmomatic/memory/", status_code=200)
def predict_trimmomatic_memory(
    file_size_bytes_1: int, file_size_bytes_2: int, format_output: bool = False
) -> dict:
    tool_name = "trimmomatic-mem"
    loaded_model = models[tool_name]

    params = {
        "file_size_bytes_1.0": file_size_bytes_1,
        "file_size_bytes_2.0": file_size_bytes_2,
    }
    print(f"params: {params}")
    df = pd.DataFrame(params, index=[0])
    print(f"df: {df}")

    y_predicted = loaded_model.predict(df)
    # y_predicted is a numpy array. Call tolist() and get the first value from list
    y_predicted_denormalized = y_predicted.tolist()[0]

    if format_output:
        y_predicted_denormalized = to_kilo_mega_giga_bytes(y_predicted_denormalized)

    return {"Tool name": tool_name, "Required memory": y_predicted_denormalized}


@api_router.get("/trimmomatic/cpu/", status_code=200)
def predict_trimmomatic_cpu(file_size_bytes_1: int, file_size_bytes_2: int) -> dict:

    result = predict_trimmomatic_memory(file_size_bytes_1, file_size_bytes_2)

    tool_name = "trimmomatic-cpu"
    loaded_model = models[tool_name]

    params = {
        "file_size_bytes_1.0": file_size_bytes_1,
        "file_size_bytes_2.0": file_size_bytes_2,
        "memory.max_usage_in_bytes": result["Required memory"],
    }
    print(f"params: {params}")
    df = pd.DataFrame(params, index=[0])
    print(f"df: {df}")

    y_predicted = loaded_model.predict(df)
    # y_predicted is a numpy array. Call tolist() and get the first value from list
    y_predicted_denormalized = y_predicted.tolist()[0]

    return {"Tool name": tool_name, "Required CPU": y_predicted_denormalized}
"""


#################
## featurecounts
#################
@api_router.get("/featurecounts/memory/", status_code=200)
def predict_featurecounts_memory(
    alignment_file_size_bytes: int = 0, format_output: bool = False
) -> dict:
    tool_name = "featurecounts-mem"
    loaded_model = models[tool_name]

    params = {
        "alignment": float(alignment_file_size_bytes),
        "anno|reference_gene_sets": float(
            0
        ),  # redundant input? Investigate why it is needed.
        "extended_parameters|genome": float(
            0
        ),  # redundant input? Investigate why it is needed.
        "reference_gene_sets": float(
            0
        ),  # redundant input? Investigate why it is needed.
    }

    # The default value for input parameters is 0
    # The model expects nan for 0 values
    for key in params:
        if params[key] == 0:
            params[key] = np.nan

    print(f"params: {params}")
    df = pd.DataFrame(params, index=[0])
    print(f"df: {df}")

    y_predicted = loaded_model.predict(df)
    # y_predicted is a numpy array. Call tolist() and get the first value from list
    y_predicted_denormalized = y_predicted.tolist()[0]

    if format_output:
        y_predicted_denormalized = to_kilo_mega_giga_bytes(y_predicted_denormalized)

    return {"Tool name": tool_name, "Required memory": y_predicted_denormalized}


@api_router.get("/featurecounts/cpu/", status_code=200)
def predict_featurecounts_cpu(alignment_file_size_bytes: int = 0) -> dict:

    result = predict_featurecounts_memory(alignment_file_size_bytes)

    tool_name = "featurecounts-cpu"
    loaded_model = models[tool_name]

    params = {
        "alignment": float(alignment_file_size_bytes),
        "anno|reference_gene_sets": float(
            0
        ),  # redundant input? Investigate why it is needed.
        "extended_parameters|genome": float(
            0
        ),  # redundant input? Investigate why it is needed.
        "reference_gene_sets": float(
            0
        ),  # redundant input? Investigate why it is needed.
        "memory.max_usage_in_bytes": result["Required memory"],
    }

    # The default value for input parameters is 0
    # The model expects nan for 0 values
    for key in params:
        if params[key] == 0:
            params[key] = np.nan

    print(f"params: {params}")
    df = pd.DataFrame(params, index=[0])
    print(f"df: {df}")

    y_predicted = loaded_model.predict(df)
    # y_predicted is a numpy array. Call tolist() and get the first value from list
    y_predicted_denormalized = y_predicted.tolist()[0]

    return {"Tool name": tool_name, "Required CPU": y_predicted_denormalized}


#############
## freebayes
#############
"""
@api_router.get("/freebayes/memory/", status_code=200)
def predict_freebayes_memory(
    file_size_bytes_1: int, file_size_bytes_2: int, format_output: bool = False
) -> dict:
    tool_name = "freebayes-mem"
    loaded_model = models[tool_name]

    params = {
        "file_size_bytes_1.0": file_size_bytes_1,
        "file_size_bytes_2.0": file_size_bytes_2,
    }
    print(f"params: {params}")
    df = pd.DataFrame(params, index=[0])
    print(f"df: {df}")

    y_predicted = loaded_model.predict(df)
    # y_predicted is a numpy array. Call tolist() and get the first value from list
    y_predicted_denormalized = y_predicted.tolist()[0]

    if format_output:
        y_predicted_denormalized = to_kilo_mega_giga_bytes(y_predicted_denormalized)

    return {"Tool name": tool_name, "Required memory": y_predicted_denormalized}


@api_router.get("/freebayes/cpu/", status_code=200)
def predict_freebayes_cpu(file_size_bytes_1: int, file_size_bytes_2: int) -> dict:

    result = predict_freebayes_memory(file_size_bytes_1, file_size_bytes_2)

    tool_name = "freebayes-cpu"
    loaded_model = models[tool_name]

    params = {
        "file_size_bytes_1.0": file_size_bytes_1,
        "file_size_bytes_2.0": file_size_bytes_2,
        "memory.max_usage_in_bytes": result["Required memory"],
    }
    print(f"params: {params}")
    df = pd.DataFrame(params, index=[0])
    print(f"df: {df}")

    y_predicted = loaded_model.predict(df)
    # y_predicted is a numpy array. Call tolist() and get the first value from list
    y_predicted_denormalized = y_predicted.tolist()[0]

    return {"Tool name": tool_name, "Required CPU": y_predicted_denormalized}
"""


app.include_router(api_router)


if __name__ == "__main__":
    # Use this for debugging purposes only
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="debug")
