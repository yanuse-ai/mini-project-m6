import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from employee_attrition_model.config.core import config
from employee_attrition_model.pipeline import employee_attrition_prediction_pipe
from employee_attrition_model.processing.data_manager import load_dataset, save_pipeline

from employee_attrition_model.processing.features import FeatureOrdinalEncoder

def run_training() -> None:
    
    """
    Train the model.
    """

    # read training data
    data = load_dataset(file_name = config.app_config_.training_data_file)
    print(data.shape)

    encoder = FeatureOrdinalEncoder(variable = config.model_config_.target)
    data = encoder.fit_transform(data)
    
    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        
        data[config.model_config_.features],     # predictors
        data[config.model_config_.target],       # target
        test_size = config.model_config_.test_size,
        random_state=config.model_config_.random_state,   # set the random seed here for reproducibility
    )
    print(f"Data: X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}")

    # Pipeline fitting
    employee_attrition_prediction_pipe.fit(X_train, y_train)
    #y_pred = employee_attrition_prediction_pipe.predict(X_test)

    # Calculate the score/error
    print("accuracy:", round(employee_attrition_prediction_pipe.score(X_test, y_test), 2))

    # persist trained model
    save_pipeline(pipeline_to_persist = employee_attrition_prediction_pipe)
    
if __name__ == "__main__":
    run_training()