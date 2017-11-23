import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from .feature import (ColumnData, Feature, CategoricalFeature,
                      OrdinalFeature, ClassTarget)

class FeatureCollection:
    def __init__(self, df):
        self.data = df
        self.feature_names = df.columns
        self.collection = {f: Feature(df[f]) \
                                 for f in self.feature_names}

class CategoricalFeatureCollection:
    def __init__(self, df):
        self.data = df
        self.feature_names = df.columns
        self.collection = {f: CategoricalFeature(df[f]) \
                                 for f in self.feature_names}

    def fuse_IndividualCategories(self, dict_new_categories):
        '''
        Combine categories for each feature in a collection. Which categories
        to fuse for each features is instructed in dict_new_categories, whose
        key is the feature name and value a list of lists. The actual fusion
        is done by fuse_categories() method of CategoricalFeature class.
        '''
        new_categoricals = self.data.copy()
        for key, val in dict_new_categories.items():
            new_categoricals[key] = self.collection[key].fuse_categories(val)
        return new_categoricals

class OrdinalFeatureCollection:
    def __init__(self, df):
        super().__init__(df)
        self.collection = {f: OrdinalFeature(df[f]) \
                                 for f in self.feature_names}
