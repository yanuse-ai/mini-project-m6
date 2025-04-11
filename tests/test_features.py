
"""
Note: These tests will fail if you have not first trained the model.
"""

import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
from employee_attrition_model.config.core import config
from employee_attrition_model.processing.features import FeatureOrdinalEncoder

def test_attrition_encoder(sample_input_data):
    # Given
    encoder = FeatureOrdinalEncoder(variable = config.model_config_.target)
    assert sample_input_data[0].loc[342, 'attrition'] == 'No'

    # When
    subject = encoder.fit(sample_input_data[0]).transform(sample_input_data[0])

    # Then
    assert subject.loc[342, 'attrition'] == 0.0

