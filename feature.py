import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.neighbors import KernelDensity

# Classes for individual features and class target
class ColumnData:
    def __init__(self, data, target=None):
        self.data = data
        self.target = target

        self.num_samples = len(self.data)
        self.unique_values = sorted(self.data.unique())
        self.num_unique_values = len(self.unique_values)
        self._counts = self.data.value_counts()
        self._frequencies = self._counts / self.num_samples

        if self.target is None:
            self._isSameSize = False
        elif self.num_samples == self.target.shape[0]:
            self._isSameSize = True
        else:
            self._isSameSize = False

class Feature(ColumnData):
    def __init__(self, feature, target=None):
        super().__init__(feature, target=target)
        self.value_counts = self._counts
        self.value_frequencies = self._frequencies

class CategoricalFeature(Feature):
    def __init__(self, feature, target=None):
        super().__init__(feature, target=target)
        # Categorical-specific attributes
        if self.num_unique_values == 2:
            self.feature_type = 'binary'
        elif self.num_unique_values > 2:
            self.feature_type = 'categorical'
        else:
            self.feature_type = 'unknown feature type'

        # Target-dependent attributes
        if self._isSameSize:
            self.contingency_table = \
                        pd.crosstab(self.data, target)
            self.frequency_table = \
                        self.contingency_table / self.num_samples
            self.conditional_probas = \
                        self.contingency_table.div(self.value_counts, axis=0)
        else:
            self.contingency_table = None
            self.frequency_table = None
            self.conditional_probas = None

    def fuse_categories(self, new_category_list):
        new_feature = self.data.copy()
        for i, list_ in enumerate(new_category_list):
            new_feature[self.data.isin(list_)] = i
        return new_feature

    def convert2proba(self, target_class=1):
        feature2proba = self.data.replace(
                                self.conditional_probas.to_dict()[target_class])
        return feature2proba

    def calculate_deviation(self, classes='all', mode='subtraction'):
        if not self._isSameSize:
            print('ERROR: Feature and Target lengths must be the same.')
            return

        table = self.conditional_probas.copy()
        for key, val in ClassTarget(self.target).class_frequencies\
                                                .to_dict().items():
            if mode == 'ratio':
                table[key] = table[key] / val
            else:
                table[key] = table[key] - val

        minmax = pd.DataFrame()
        minmax['max'] = table.max(axis=0)
        minmax['min'] = table.min(axis=0)

        return table, minmax

class OrdinalFeature(Feature):
    def __init__(self, feature):
        super().__init__(feature)
        self.feature_type = 'ordinal'
        self.max_value = np.max(self.data)
        self.min_value = np.min(self.data)
        self.bulk_kde = None
        self.target_kde = None

    def statsmodelsKDE(self, graph=True, ax=None, normed=True,
                   hist_bins='auto', color='skyblue', alpha=0.5,
                   kernel='gau', bw='normal_reference', fft=True,
                   weights=None, gridsize=None, adjust=1, cut=3,
                   clip=(-np.inf, np.inf)):
        '''
        KDE with statsmodels nonparametric estimator.
        '''
        kde = sm.nonparametric.KDEUnivariate(self.data.astype('float'))
        kde.fit(kernel=kernel, bw=bw, fft=fft, weights=weights,
                gridsize=gridsize, adjust=adjust, cut=cut, clip=clip)

        if hist_bins == 'auto':
            bins = self.num_unique_values
        elif type(hist_bins) == 'int':
            bins = hist_bins
        elif type(hist_bins == 'float'):
            bins = int(hist_bins)
        else:
            print('Error: bins must be an integer')

        if graph:
            if ax is None:
                fig, ax = plt.subplots(1, 1, figsize=(5, 5))

            if normed:
                ax.hist(self.data, bins=bins, normed=True,
                                                alpha=alpha, color=color)
                ax.plot(kde.support, kde.density, ls='--', color=color)
            else:

                ax.hist(self.data, bins=bins, normed=False,
                        alpha=alpha, color=color)
                ax.plot(kde.support,
                        self.num_samples*kde.density,
                        ls='--', color=color)
        return kde

    def sklearnKDE(self, graph=True, ax=None,
                 span='auto', bandwidth=1, algorithm='auto', kernel='gaussian',
                 metric='euclidean', atol=0, rtol=0, breadth_first=True,
                 leaf_size=40, metric_params=None,
                 hist_bins='auto', color='skyblue', alpha=0.5):
        '''
        KDE with sklearn estimator.
        '''
        feature = self.data

        kde = KernelDensity(bandwidth=bandwidth, algorithm=algorithm,
                            kernel=kernel, metric=metric, atol=atol,
                            rtol=rtol, breadth_first=breadth_first,
                            leaf_size=leaf_size, metric_params=metric_params)
        kde.fit(np.array(feature).reshape(-1, 1))


        if hist_bins == 'auto':
            bins = len(feature.unique())
        elif type(hist_bins) == 'int':
            bins = hist_bins
        elif type(hist_bins == 'float'):
            bins = int(hist_bins)
        else:
            print('Error: bins must be an integer')

        if span == 'auto':
            span = np.linspace(np.min(feature.unique()),
                               np.max(feature.unique()), 30).reshape(-1, 1)
        if graph:
            if ax is None:
                fig, ax = plt.subplots(1, 1, figsize=(5, 5))

            ax.hist(feature, bins=bins, normed=True,
                             color=color, alpha=alpha, label=None)
            ax.plot(span, np.exp(kde.score_samples(span)),
                             color=color, alpha=alpha, ls='--', label='Bulk')

        return kde

    def find_nearest_in_list(self, list_):
        '''
        For each entry in the feature (which is a single column),
        replace the original value with its nearest value in the list_.
        This is a utility function for KDE method to reduce number of
        processing.
        '''
        nearest = []
        for i in range(self.num_samples):
            idx = (np.abs(list_- self.data[i])).argmin()
            nearest.append(list_[idx])
        return pd.Series(nearest)

    def calculateDensityRatio(self, target, span, mode='percent',
                   kernel='gau', bw='normal_reference', fft=True,
                   weights=None, gridsize=None, adjust=1, cut=3,
                   clip=(-np.inf, np.inf)):
        '''
        Convert the feature space into space of gain in conditional probability
        for each value of the feature. It assumes that one is looking at the
        conditional probability of being in class1 (i.e. target == 1).

        statsmodel version of KDE used.
        ** For now, this can only be used to calculate conditional probability
        of being in class1 in a binary classification task. To be updated to
        multiclass label.
        '''
        if self.num_samples != len(target):
            print('Error: Target size must be the same as the feature size.')
            return

        class1 = OrdinalFeature(self.data[target == 1])
        bulk_size = len(target)
        class_size = np.sum(target)
        class1_freq = class_size / bulk_size

        self.bulk_kde = self.statsmodelsKDE(bw=bw, graph=False)
        self.target_kde = class1.statsmodelsKDE(bw=bw, graph=False)

        bulk_dens = bulk_size*self.bulk_kde.evaluate(np.array(span))
        class_dens = class_size*self.target_kde.evaluate(np.array(span))

        ratio = class_dens/bulk_dens

        return ratio

    def convert2DensityRatio(self, target, span, mode='percent',
                   kernel='gau', bw='normal_reference', fft=True,
                   weights=None, gridsize=None, adjust=1, cut=3,
                   clip=(-np.inf, np.inf)):
        if self.num_samples != len(target):
            print('Error: Target size must be the same as the feature size.')
            return

        ratio = self.calculateDensityRatio(target, span, mode=mode,
                               kernel=kernel, bw=bw, fft=fft,
                               weights=weights, gridsize=gridsize,
                               adjust=adjust, cut=cut,
                               clip=clip)
        conversion_dict = dict(zip(span, ratio))
        converted = self.find_nearest_in_list(span).replace(conversion_dict)
        return converted

    def convert2gain(self, target, span, mode='percent',
                   kernel='gau', bw='normal_reference', fft=True,
                   weights=None, gridsize=None, adjust=1, cut=3,
                   clip=(-np.inf, np.inf)):
        '''
        Convert the feature space into space of gain in conditional probability
        for each value of the feature. It assumes that one is looking at the
        conditional probability of being in class1 (i.e. target == 1).

        statsmodel version of KDE used.
        '''
        if self.num_samples != len(target):
            print('Error: Target size must be the same as the feature size.')
            return

        ratio = self.calculateDensityRatio(target, span, mode=mode,
                               kernel=kernel, bw=bw, fft=fft,
                               weights=weights, gridsize=gridsize,
                               adjust=adjust, cut=cut,
                               clip=clip)

        if mode == 'fraction':
            gain = (ratio/class1_freq) - 1
        else:
            gain = 100*((ratio/class1_freq) - 1)
        conversion_dict = dict(zip(span, gain))

        converted = self.find_nearest_in_list(span).replace(conversion_dict)

        return converted

class ClassTarget(ColumnData):
    def __init__(self, target):
        super().__init__(target)
        self.class_counts = self._counts
        self.class_frequencies = self._frequencies
