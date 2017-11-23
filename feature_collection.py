import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from feature import (ColumnData, Feature, CategoricalFeature,
                      OrdinalFeature, ClassTarget)

class FeatureCollection:
    def __init__(self, df, target=None):
        self.data = df
        self.feature_names = df.columns
        self.num_samples = len(self.data)
        self.target = target
        if self.target is None:
            self._isSameSize = False
        elif self.num_samples == self.target.shape[0]:
            self._isSameSize = True
        else:
            self._isSameSize = False

class CategoricalFeatureCollection(FeatureCollection):
    '''
    Collection of CategoricalFeature objects.
    '''
    def __init__(self, df, target=None):
        super().__init__(df, target=target)
        self.collection = {
                feature_name: \
                    CategoricalFeature(df[feature_name], target=self.target) \
                for feature_name in self.feature_names}

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

    def convert2CondProba(self, target_class=1):
        if not self._isSameSize:
            print('ERROR: Feature and Target lengths must be the same.')
            return

        proba_space = pd.DataFrame()
        for i, feature_name in enumerate(self.feature_names):
            F = self.collection[feature_name]
            proba_space[feature_name] = \
                                F.convert2CondProba(target_class=target_class)

        return proba_space

class OrdinalFeatureCollection(FeatureCollection):
    def __init__(self, df, target=None):
        super().__init__(df, target=target)
        self.collection = {
                feature_name: \
                    OrdinalFeature(df[feature_name], target=self.target) \
                for feature_name in self.feature_names}

    def convert2CondProba(self, target_class=1):
        if not self._isSameSize:
            print('ERROR: Feature and Target lengths must be the same.')
            return

        proba_space = pd.DataFrame()
        for i, feature_name in enumerate(self.feature_names):
            F = self.collection[feature_name]
            proba_space[feature_name] = \
                                F.convert2CondProba()

        return proba_space
