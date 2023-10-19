import pandas as pd
import numpy as np
import subprocess
import sys
subprocess.call([sys.executable, '-m', 'pip', 'install', 'category_encoders'])  #replaces !pip install
import category_encoders as ce
import sklearn
sklearn.set_config(transform_output="pandas")  #pass pandas tables through pipeline instead of numpy matrices
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


titanic_variance_based_split = 107
customer_variance_based_split = 113

class CustomMappingTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, mapping_column, mapping_dict:dict):
    assert isinstance(mapping_dict, dict), f'{self.__class__.__name__} constructor expected dictionary but got {type(mapping_dict)} instead.'
    self.mapping_dict = mapping_dict
    self.mapping_column = mapping_column

  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return self

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
    assert self.mapping_column in X.columns.to_list(), f'{self.__class__.__name__}.transform unknown column "{self.mapping_column}"'  #column legit?

    #Set up for producing warnings. First have to rework nan values to allow set operations to work.
    #In particular, the conversion of a column to a Series, e.g., X[self.mapping_column], transforms nan values in strange ways that screw up set differencing.
    #Strategy is to convert empty values to a string then the string back to np.nan
    placeholder = "NaN"
    column_values = X[self.mapping_column].fillna(placeholder).tolist()  #Convert all nan values to the string "NaN" in new list
    column_values = [np.nan if v == placeholder else v for v in column_values]  #Now convert back to np.nan
    keys_values = self.mapping_dict.keys()

    column_set = set(column_values)  #Without the conversion above, the set will fail to have np.nan values where they should be
    keys_set = set(keys_values)      #This will have np.nan values where they should be so no conversion necessary

    #Verify all keys are contained in column
    keys_not_found = keys_set - column_set
    if keys_not_found:
      print(f"\nWarning: {self.__class__.__name__}[{self.mapping_column}] these mapping keys do not appear in the column: {keys_not_found}\n")

    #Verify if keys are absent
    keys_absent = column_set -  keys_set
    if keys_absent:
      print(f"\nWarning: {self.__class__.__name__}[{self.mapping_column}] these values in the column do not contain corresponding mapping keys: {keys_absent}\n")

    #Actual mapping
    X_ = X.copy()
    X_[self.mapping_column].replace(self.mapping_dict, inplace=True)
    return X_

  def fit_transform(self, X, y = None):
    #self.fit(X,y)
    result = self.transform(X)
    return result

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

class CustomSigma3Transformer(BaseEstimator, TransformerMixin):
  def __init__(self, target_column):
    self.target_column = target_column
    self.three_sigma = None

  def fit(self, X, y = None):
    assert isinstance(X, pd.core.frame.DataFrame), f'expected Dataframe but got {type(X)} instead.'
    assert self.target_column in X.columns, f'unknown column {self.target_column}'
    assert all([isinstance(v, (int, float)) for v in X[self.target_column].to_list()])

    mean = X[self.target_column].mean()
    std_dev = X[self.target_column].std()
    self.three_sigma = (mean-3*std_dev, mean+3*std_dev)

  def transform(self, X):
    assert self.three_sigma is not None, f'NotFittedError: This {self.__class__.__name__} instance is not fitted yet. Call "fit" with appropriate arguments before using this estimator.'
    transformed_df = X.copy()
    transformed_df[self.target_column] = transformed_df[self.target_column].clip(lower=self.three_sigma[0], upper=self.three_sigma[1])
    df_clean = transformed_df.reset_index(drop=True)
    return df_clean

  def fit_transform(self, X, y = None):
    self.fit(X,y)
    result = self.transform(X)
    return result
  
class CustomTukeyTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, target_column, fence='outer'):
    assert fence in ['inner', 'outer']
    self.target_column = target_column
    self.fence = fence
    self.boundaries = None

  def fit(self, X, y = None):
    assert isinstance(X, pd.core.frame.DataFrame), f'expected Dataframe but got {type(X)} instead.'
    assert self.target_column in X.columns, f'unknown column {self.target_column}'
    assert all([isinstance(v, (int, float)) for v in X[self.target_column].to_list()])

    q1 = X[self.target_column].quantile(0.25)
    q3 = X[self.target_column].quantile(0.75)

    iqr = q3-q1
    fence_values = {'outer': 3, 'inner': 1.5}
    fence_multiplier = fence_values.get(self.fence)

    low = q1 - fence_multiplier * iqr
    high = q3 + fence_multiplier * iqr

    self.boundaries = (low, high)
    return self

  def transform(self, X):
    assert self.boundaries is not None, f'NotFittedError: This {self.__class__.__name__} instance is not fitted yet. Call "fit" with appropriate arguments before using this estimator.'
    transformed_df = X.copy()
    transformed_df[self.target_column] = transformed_df[self.target_column].clip(lower=self.boundaries[0], upper=self.boundaries[1])
    df_clean = transformed_df.reset_index(drop=True)
    return df_clean


  def fit_transform(self, X, y = None):
    self.fit(X,y)
    result = self.transform(X)
    return result

class CustomRobustTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, column):
    self.target_column = column
    self.iqr = None
    self.median = None

  def fit(self, X, y = None):
    assert isinstance(X, pd.core.frame.DataFrame), f'expected Dataframe but got {type(X)} instead.'
    assert self.target_column in X.columns, f'unknown column {self.target_column}'
    assert all([isinstance(v, (int, float)) for v in X[self.target_column].to_list()])

    self.iqr = X[self.target_column].quantile(.75) - X[self.target_column].quantile(.25)
    self.median = X[self.target_column].median()

    return self

  def transform(self, X):
    assert self.iqr is not None and self.median is not None, f'NotFittedError: This {self.__class__.__name__} instance is not fitted yet. Call "fit" with appropriate arguments before using this estimator.'

    transformed_df = X.copy()
    transformed_df[self.target_column] -= self.median
    transformed_df[self.target_column] /= self.iqr
    return transformed_df

  def fit_transform(self, X, y = None):
    self.fit(X,y)
    result = self.transform(X)
    return result
  
def find_random_state(features_df, labels, n=200):
  model = KNeighborsClassifier(n_neighbors=5)  #instantiate with k=5.
  var = []  #collect test_error/train_error where error based on F1 score

  for i in range(1, n):
    train_X, test_X, train_y, test_y = train_test_split(features_df, labels, test_size=0.2, shuffle=True,
                                                    random_state=i, stratify=labels)
    model.fit(train_X, train_y)  #train model
    train_pred = model.predict(train_X)           #predict against training set
    test_pred = model.predict(test_X)             #predict against test set
    train_f1 = f1_score(train_y, train_pred)   #F1 on training predictions
    test_f1 = f1_score(test_y, test_pred)      #F1 on test predictions
    f1_ratio = test_f1/train_f1          #take the ratio
    var.append(f1_ratio)
  rs_value = sum(var)/len(var)  #get average ratio value    
  idx = np.array(abs(var - rs_value)).argmin()  #find the index of the smallest value
  return idx
