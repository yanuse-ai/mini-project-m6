from typing import List
import sys
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder

class PrintAllFeatures(BaseEstimator, TransformerMixin):
    """ Ordinal categorical variable encoder """
    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y: pd.Series = None):        
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        #print(X.head())

        return X

class RemoveTarget(BaseEstimator, TransformerMixin):
    def __init__(self, variable:str):

        if not isinstance(variable, str):
            raise ValueError("variable name should be a str")
        self.variable = variable

    def fit(self, X: pd.DataFrame, y: pd.Series = None):        
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        
        X.drop(self.variable, axis=1, inplace=True)

        return X

class FeatureOrdinalEncoder(BaseEstimator, TransformerMixin):
    """ Ordinal categorical variable encoder """
    def __init__(self, variable:str):

        if not isinstance(variable, str):
            raise ValueError("variable name should be a str")
        #print(f"{variable}") 
        self.variable = variable
        self.encoder = OrdinalEncoder()

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        ##X = X.copy()
        ##self.encoder.fit(X[[self.variable]])
        # Get encoded feature names
        ##self.encoded_features_names = self.encoder.get_feature_names_out(self.variable)
        ##print(self.encoded_features_names)
        
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        X[[self.variable]] = self.encoder.fit_transform(X[[self.variable]])

        #print(f"{self.variable} done")       

        return X