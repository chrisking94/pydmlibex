from .dataprocessor import *
from .transformer import TsfmFunction


class Normalizer(RSDataProcessor):
    def __init__(self, features2process, name='', forecolor='black'):
        RSDataProcessor.__init__(self, features2process, name, forecolor, 'pink', 'highlight')

    @abstractmethod
    def _fit(self, X, y):
        self.error('Not implemented!')

    @abstractmethod
    def _transform(self, X):
        self.error('Not implemented!')


class FeatureNormalizer(Normalizer):
    def __init__(self, features2process, name=''):
        Normalizer.__init__(self, features2process, name, 'blue')

    @abstractmethod
    def _fit(self, X, y):
        self.error('Not implemented!')

    @abstractmethod
    def _transform(self, X):
        self.error('Not implemented!')


class FNMinMax(FeatureNormalizer):
    def __init__(self, features2process):
        FeatureNormalizer.__init__(self, features2process, 'minmax')
        self.range = None
        self.min = None

    def _fit(self, X, y):
        self.range = X.max() - X.min()
        self.min = X.min()

    def _transform(self, X):
        return (X - self.min) / self.range


class FNAtan(FeatureNormalizer):
    def __init__(self, features2process):
        FeatureNormalizer.__init__(self, features2process, 'atan')

    def _fit(self, X, y):
        pass

    def _transform(self, X):
        return np.emath.arctanh(X) * 2 / np.pi


class FNZScore(FeatureNormalizer):
    def __init__(self, features2process):
        FeatureNormalizer.__init__(self, features2process, 'zscore')
        self.mean = None
        self.std = None

    def _fit(self, X, y):
        self.mean = X.mean()
        self.std = X.std()

    def _transform(self, X):
        return (X - self.mean) / self.std

