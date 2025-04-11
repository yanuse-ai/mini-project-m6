import os
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import typing as t
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from employee_attrition_prediction_model import __version__ as _version
from employee_attrition_prediction_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config


##  Pre-Pipeline Preparation

# handle outliers
def handle_outliers(dataframe: pd.DataFrame):

    df_data_filtered = dataframe.copy()
    
    for col in list(df_data_filtered.select_dtypes(include=['int64']).columns):
        q1 = df_data_filtered[col].quantile(0.25)
        q3 = df_data_filtered[col].quantile(0.75)
        IQR = q3 - q1

        lower_bound = q1 - 1.5 * IQR
        upper_bound = q3 + 1.5 * IQR

        #print(f"{q1}=>{lower_bound}, {q3}=>{upper_bound}, {IQR}")
        f1 = df_data_filtered[col] >= lower_bound
        f2 = df_data_filtered[col] <= upper_bound
        df_data_filtered = df_data_filtered[f1 & f2]
    
    return df_data_filtered

def ordinal_encoder(dataframe: pd.DataFrame):
    df_data_part_b = dataframe.copy()
    
    cat_col = []
    for col in list(df_data_part_b.select_dtypes(include=['object']).columns):
        cat_col.append(col)
    
    ordinal_enc = OrdinalEncoder()
    df_data_part_b[cat_col + ['attrition']] = ordinal_enc.fit_transform(df_data_part_b[cat_col + ['attrition']])

    return df_data_part_b

def pre_pipeline_preparation(*, data_frame: pd.DataFrame) -> pd.DataFrame:

    # handle the outliers
    data_frame = handle_outliers(data_frame)

    return data_frame

def load_dataset(*, file_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    print(f"Original data: {dataframe.shape}")
    transformed = pre_pipeline_preparation(data_frame = dataframe)
    #transformed = ordinal_encoder(transformed)
    print(f"After transform data: {transformed.shape}")
    return transformed


def save_pipeline(*, pipeline_to_persist: Pipeline) -> None:
    """Persist the pipeline.
    Saves the versioned model, and overwrites any previous saved models. 
    This ensures that when the package is published, there is only one trained model that 
    can be called, and we know exactly how it was built.
    """

    # Prepare versioned save file name
    save_file_name = f"{config.app_config_.pipeline_save_file}{_version}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name

    remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)
    print("Model/pipeline saved successfully.")


def load_pipeline(*, file_name: str) -> Pipeline:
    """Load a persisted pipeline."""

    file_path = TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model


def remove_old_pipelines(*, files_to_keep: t.List[str]) -> None:
    """
    Remove old model pipelines.
    This is to ensure there is a simple one-to-one mapping between the package version and 
    the model version to be imported and used by other applications.
    """
    do_not_delete = files_to_keep + ["__init__.py"]
    if not os.path.exists(TRAINED_MODEL_DIR):
        os.mkdir(TRAINED_MODEL_DIR)

    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()
