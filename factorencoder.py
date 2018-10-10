from .dataprocessor import *


class FactorEncoder(RSDataProcessor):
    def __init__(self, features2process, name=''):
        RSDataProcessor.__init__(self, features2process, name, 'white', 'black')

    def _encode(self, data, features, label):
        self.error('Not implemented!')

    @abstractmethod
    def _fit(self, X, y):
        raise NotImplementedError()

    @abstractmethod
    def _transform(self, X):
        raise NotImplementedError()


class FEOneHot(FactorEncoder):
    def __init__(self, features2process):
        FactorEncoder.__init__(self, features2process)
        self.encoder = None
        self.val_list = []

    def _fit(self, X, y):
        X = X.astype('str', copy=False)
        for col in X.columns:
            self.val_list.append((col, np.unique(X[col])))

    def _transform(self, X):
        X = X.astype('str', copy=False)
        X_ = RSData()
        for col_vals in self.val_list:
            col = col_vals[0]
            for val in col_vals[1]:
                new_col = '%s_%s' % (col, val)
                X_[new_col] = X[col] == val
        return X_.astype('int64', copy=False)
