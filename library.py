import pandas as pd
import numpy as np
import sklearn
sklearn.set_config(transform_output="pandas")  #pass pandas tables through pipeline instead of numpy matrices
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

#This class will rename one or more columns.
class CustomRenamingTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, rename_dict:dict):
    assert isinstance(rename_dict, dict), f'{self.__class__.__name__} constructor expected dictionary for rename_dict but got {type(rename_dict)} instead.'
    self.rename_dict = rename_dict

  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return self

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
    missing_columns = [key for key in self.rename_dict.keys() if key not in X.columns.to_list()]
    if missing_columns:
        raise AssertionError(f'{self.__class__.__name__}: cannot rename unknown columns: {missing_columns}')

    X_ = X.copy()
    X_.rename(columns=self.rename_dict, inplace=True)
    return X_

  def fit_transform(self, X, y = None):
    #self.fit(X,y)
    result = self.transform(X)
    return result

#This class will perform a One Hot Encoding
class CustomOHETransformer(BaseEstimator, TransformerMixin):
  def __init__(self, target_column, dummy_na=False, drop_first=False):
    assert isinstance(target_column, str), f'{self.__class__.__name__} constructor expected str for target_column but got {type(target_column)} instead.'
    assert isinstance(dummy_na, bool), f'{self.__class__.__name__} constructor expected boolean for dummy_na but got {type(dummy_na)} instead.'
    assert isinstance(drop_first, bool), f'{self.__class__.__name__} constructor expected boolean for drop_first but got {type(drop_first)} instead.'
    self.target_column = target_column
    self.dummy_na = dummy_na
    self.drop_first = drop_first

  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return self

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
    assert self.target_column in X.columns.to_list(), f'{self.__class__.__name__}.transform unknown column "{self.target_column}"'

    X_ = X.copy()
    X_return = pd.get_dummies(X_,
                              prefix=self.target_column,
                              prefix_sep='_',
                              columns=[f'{self.target_column}'],
                              dummy_na=self.dummy_na,
                              drop_first=self.drop_first
                              )
    return X_return

  def fit_transform(self, X, y = None):
    #self.fit(X,y)
    result = self.transform(X)
    return result

