from typing import List
import sys
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder


class FeatureOrdinalEncoder(BaseEstimator, TransformerMixin):
    """ Ordinal categorical variable encoder """
    def __init__(self, variable:str):

        if not isinstance(variable, str):
            raise ValueError("variable name should be a str")

        self.variable = variable
        self.encoder = OrdinalEncoder()

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        X = X.copy()
        self.encoder.fit(X[[self.variable]])
        # Get encoded feature names
        self.encoded_features_names = self.encoder.get_feature_names_out(self.variable)
        
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        
        encoded_features = self.encoder.transform(X[[self.variable]])
        # Append encoded features to X
        X[self.encoded_features_names] = encoded_features

        ## drop 'weekday' column after encoding
        ##X.drop(self.variable, axis=1, inplace=True)
        print(X.shape)       

        return X