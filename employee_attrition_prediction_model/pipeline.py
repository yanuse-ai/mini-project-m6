import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings("ignore")

from employee_attrition_prediction_model.config.core import config
from employee_attrition_prediction_model.processing.features import FeatureOrdinalEncoder, PrintAllFeatures, RemoveTarget

employee_attrition_prediction_pipe = Pipeline([

    ######## Handle outliers ########
    #('handle_outliers_temp', OutlierHandler(variable = config.model_config_.temp_var)),
 
    ######## One-hot encoding ########
    ('onehot_encoder_businesstravel', FeatureOrdinalEncoder(variable = config.model_config_.businesstravel_var)),
    ('onehot_encoder_department', FeatureOrdinalEncoder(variable = config.model_config_.department_var)),
    ('onehot_encoder_educationfield', FeatureOrdinalEncoder(variable = config.model_config_.educationfield_var)),
    ('onehot_encoder_gender', FeatureOrdinalEncoder(variable = config.model_config_.gender_var)),
    ('onehot_encoder_jobrole', FeatureOrdinalEncoder(variable = config.model_config_.jobrole_var)),
    ('onehot_encoder_maritalstatus', FeatureOrdinalEncoder(variable = config.model_config_.maritalstatus_var)),
    ('onehot_encoder_overtime', FeatureOrdinalEncoder(variable = config.model_config_.overtime_var)),
    ('onehot_encoder_over18', FeatureOrdinalEncoder(variable = config.model_config_.over18_var)),
    #('onehot_encoder_target', FeatureOrdinalEncoder(variable = config.model_config_.target)),

    #('remove_target', RemoveTarget(variable = config.model_config_.target)),
    ('print_all_features', PrintAllFeatures()),

    # Classifier
    ('model_rf', CatBoostClassifier(
            iterations=config.model_config_.iterations,
            learning_rate=config.model_config_.learning_rate,
            loss_function=config.model_config_.loss_function))
    
    ])
