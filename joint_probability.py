import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from feature import (ColumnData, Feature,
                     CategoricalFeature, OrdinalFeature, ClassTarget)
from feature_collection import (FeatureCollection,
                                CategoricalFeatureCollection,
                                OrdinalFeatureCollection)                     
from itertools import combinations
from ipyparallel import Client

class JointProbability:
    def __init__(self):
        self.contingency = None

    def fit(self, feature, target, label=None):
        '''
        id_ : (n_samples,)
            column vector containing ids for each sample
        feature: (n_samples, n_features)
            features to be joined
        target : (n_sampes, )
            column vector containing class label of each sample
        '''
        if label is None:
            label = 'cond_proba'
        else:
            label = label

        df1 = pd.DataFrame()
        df1['combined_feature'] = feature.apply(tuple, axis=1)

        contingency = pd.crosstab(df1['combined_feature'], target)
        contingency[label] = contingency[1]/(contingency[0] + contingency[1])
        self.contingency = contingency.reset_index()

        return

    def transform(self, feature, label=None):
        if label is None:
            label = 'cond_proba'
        else:
            label = label

        df1 = pd.DataFrame()
        df1['combined_feature'] = feature.apply(tuple, axis=1)

        if self.contingency is None:
            print('Error: Object has to be fit first')
            return

        df2 = df1.merge(self.contingency[['combined_feature', label]], how='left',
                       left_on='combined_feature', right_on='combined_feature')

        return df2[label]

    def fit_transform(self, feature, target, label=None):
        '''
        id_ : (n_samples,)
            column vector containing ids for each sample
        feature: (n_samples, n_features)
            features to be joined
        target : (n_sampes, )
            column vector containing class label of each sample
        '''
        if label is None:
            label = 'cond_proba'
        else:
            label = label

        df1 = pd.DataFrame()
        df1['combined_feature'] = feature.apply(tuple, axis=1)

        contingency = pd.crosstab(df1['combined_feature'], target)
        contingency[label] = contingency[1]/(contingency[0] + contingency[1])
        self.contingency = contingency.reset_index()

        df2 = df1.merge(self.contingency[['combined_feature', label]], how='left',
                   left_on='combined_feature', right_on='combined_feature')

        return df2[label]

class JointProbabilityOptimizer:
    '''
    Use expected value of being in class1 as a criterion
    to decide whether to retain the feature or not.
    '''
    def __init__(self, feature, num_elimination=1, verbose=False):
        self.feature_list_ = list(feature.columns)
        self.num_features = len(self.feature_list_)
        self.num_samples = len(feature)
        self.best_feature_list_ = feature.columns
        self.data = feature
        #self.scores_ = {}
        self.verbose = verbose
        self.best_combination_ = None
        self.best_expected_value_ = 0

    def selectByElimination(self, target):
        '''
        score_log:
            key - name of feature removed
            val - expected value of class1 withou the key feature
        '''
        expected_vals = {}
        score_log = {}
        # Calculate the score without feature elimination
        if self.verbose:
            print(' 0/%2d: Processing Original' % (self.num_features))
        jp_initial = JointProbability()
        jp_initial.fit(self.data, target)
        contingency_initial = jp_initial.contingency
        E_initial = np.sum(contingency_initial[1] \
                           *contingency_initial['cond_proba'])
        expected_vals['initial'] = E_initial
        score_log['initial'] = 0

        for i, f in enumerate(self.feature_list_):
            if self.verbose:
                print('%2d/%2d: Processing without %s' \
                                        % ((i+1), self.num_features, f))

            # Create new feature without f
            test_list = self.feature_list_.copy()
            test_list.remove(f)

            test_feature = self.data[test_list]
            jp = JointProbability()
            jp.fit(test_feature, target)
            contingency = jp.contingency
            E = np.sum(contingency[1] * contingency['cond_proba'])
            expected_vals[f] = E
            score_log[f] = 1 - (E / E_initial)

        return expected_vals, score_log

    def selectByAddition(self, new_features, target):
        expected_vals = {}
        score_log = {}
        new_feature_list = new_features.columns
        num_features_tested = len(new_feature_list)

        # Calculate the score without feature elimination
        if self.verbose:
            print(' 0/%2d: Processing Original' % (num_features_tested))
        jp_initial = JointProbability()
        jp_initial.fit(self.data, target)
        contingency_initial = jp_initial.contingency
        E_initial = np.sum(contingency_initial[1] \
                           *contingency_initial['cond_proba'])
        expected_vals['initial'] = E_initial
        score_log['initial'] = 0

        for i, f in enumerate(new_feature_list):
            if self.verbose:
                print('%2d/%2d: Processing with %s' \
                                        % ((i+1), num_features_tested, f))

            # Create new feature without f
            test_feature = pd.concat([self.data.copy(),
                                      new_features[f]],
                                      axis=1)
            jp = JointProbability()
            jp.fit(test_feature, target)
            contingency = jp.contingency
            E = np.sum(contingency[1] * contingency['cond_proba'])
            expected_vals[f] = E
            score_log[f] = (E / E_initial) - 1

        return expected_vals, score_log

    def getExp(self, combo, target):
        test_feature = self.data[list(combo)]
        jp = JointProbability()
        jp.fit(test_feature, target)
        contingency = jp.contingency
        E = np.sum(contingency[1] * contingency['cond_proba'])
        return E

    def combinatorialSelection(self, target, N=2, parallel=False):
        '''
        score_log:
            key - name of feature removed
            val - expected value of class1 withou the key feature
        This works with N=1, where expected vals with each feature
        is calculated.
        '''
        if N > self.num_features:
            print('Error: N has to be smaller than num_features')
            return

        feature_combo = list(combinations(self.feature_list_, N))
        expected_vals = {}

        for i, combo in enumerate(feature_combo):
            E = self.getExp(combo, target)
            expected_vals[combo] = E
            if E > self.best_expected_value_:
                self.best_expected_value_ = E
                self.best_combination_ = combo

            if self.verbose:
                print('%2d/%2d: Processed'  % ((i+1), len(feature_combo)),
                      combo, 'E=%.2f' % E)

        return expected_vals

    def parallel_combinatorialSelection(self, target, N=2, n_jobs=None, c=None):
        '''
        score_log:
            key - name of feature removed
            val - expected value of class1 withou the key feature
        This works with N=1, where expected vals with each feature
        is calculated.
        '''
        if N > self.num_features:
            print('Error: N has to be smaller than num_features')
            return

        def getExp2(combo):
            test_feature = self.data[list(combo)]
            jp = JointProbability()
            jp.fit(test_feature, target)
            contingency = jp.contingency
            E = np.sum(contingency[1] * contingency['cond_proba'])
            return E

        # Set up parallel calculation
        if c is None:
            c = Client(profile="default")
        else:
            c = c

        feature_combo = list(combinations(self.feature_list_, N))

        if n_jobs is None:
            vals = c[:].map(getExp2, feature_combo)
        else:
            vals = c[:n_jobs].map(getExp2, feature_combo)
        c.wait(vals)

        expected_vals = dict(zip(feature_combo, vals.get()))

        self.best_expected_value_ = np.max(vals.get())
        best_ind = np.argmax(vals.get())
        self.best_combination_ = feature_combo[best_ind]

        return expected_vals

    def exhaustiveCombinatorialSelection(self, target, parallel=False):

        result = {}
        best_E = 0
        best_combo = 0
        if parallel:
            c = Client(profile="default")

        for i in range(1, self.num_features + 1):
            key = str(i) + '_feature_combination'
            if self.verbose:
                print('==== %s ====' % key)

            if parallel:
                vals = self.parallel_combinatorialSelection(target, N=i, c=c)
                result[key] = vals

                if self.best_expected_value_ > best_E:
                    best_E = self.best_expected_value_
                    best_combo = self.best_combination_

                if self.verbose:
                    print(vals)

            else:
                val = self.combinatorialSelection(target, N=i)
                result[key] = val

        return result

class ordinal2probability:
    def __init__(self, span):
        self.conversion_dict = None
        self.span = span

    def fit(self, feature, target, bw='normal_reference'):
        F = OrdinalFeature(feature)
        if F.num_samples != len(target):
            print('Error: Target size must be the same as the feature size.')
            return

        class1 = OrdinalFeature(F.data[target == 1])
        bulk_size = len(target)
        class_size = np.sum(target)
        class1_freq = class_size / bulk_size

        kde_bulk = F.estimateKD(bw=bw, graph=False)
        kde_class1 = class1.estimateKD(bw=bw, graph=False)

        bulk_dens = bulk_size*kde_bulk.evaluate(np.array(self.span))
        class_dens = class_size*kde_class1.evaluate(np.array(self.span))
        proba = class_dens/bulk_dens
        self.conversion_dict = dict(zip(self.span, proba))

        return

    def transform(self, feature):
        F = OrdinalFeature(feature)
        converted = F.find_nearest_in_list(self.span)\
                     .replace(self.conversion_dict)
        return converted
