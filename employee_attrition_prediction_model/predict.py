import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Union
import pandas as pd
import numpy as np

from employee_attrition_prediction_model import __version__ as _version
from employee_attrition_prediction_model.config.core import config
from employee_attrition_prediction_model.processing.data_manager import load_pipeline
from employee_attrition_prediction_model.processing.data_manager import pre_pipeline_preparation
from employee_attrition_prediction_model.processing.validation import validate_inputs


pipeline_file_name = f"{config.app_config_.pipeline_save_file}{_version}.pkl"
employee_attrition_prediction_pipe = load_pipeline(file_name = pipeline_file_name)


def make_prediction(*, input_data: Union[pd.DataFrame, dict]) -> dict:
    """Make a prediction using a saved model """
    
    validated_data, errors = validate_inputs(input_df = pd.DataFrame(input_data))
    
    validated_data = validated_data.reindex(columns = config.model_config_.features)
    
    results = {"predictions": None, "version": _version, "errors": errors}
      
    if not errors:
        predictions = employee_attrition_prediction_pipe.predict(validated_data)
        results = {"predictions": np.floor(predictions), "version": _version, "errors": errors}
    
    print(results)

    return results



if __name__ == "__main__":

    data_in = {'age': [37],
            'businesstravel': ['Travel_Rarely'],
            'dailyrate': [1373],
            'department': ['Research & Development'],
            'distancefromhome': [2],
            'education': [1],
            'educationfield': ['Life Sciences'],
            'employeecount': [1],
            'employeenumber': [4],
            'environmentsatisfaction': [4],
            'gender': ['Male'],
            'hourlyrate': [92],
            'jobinvolvement': [2],
            'joblevel': [1],
            'jobrole': ['Research Scientist'],
            'jobsatisfaction': [3],
            'maritalstatus': ['Married'],
            'monthlyincome': [2090],
            'monthlyrate': [2396],
            'numcompaniesworked': [6],
            'over18': ['Y'],
            'overtime': ['No'],
            'percentsalaryhike': [15],
            'performancerating': [3],
            'relationshipsatisfaction': [2],
            'standardhours': [80],
            'stockoptionlevel': [0],
            'totalworkingyears': [7],
            'trainingtimeslastyear': [3],
            'worklifebalance': [3],
            'yearsatcompany': [0],
            'yearsincurrentrole': [0],
            'yearssincelastpromotion': [0],
            'yearswithcurrmanager': [0],
            'attrition': [0]}

    make_prediction(input_data = data_in)