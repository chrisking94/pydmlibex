from .dataprocessor import *
from .base import np


class Reporter(RSDataProcessor):
    def __init__(self, features2process, name='Reporter', forecolor='blue'):
        RSDataProcessor.__init__(self, features2process, name, forecolor, 'black', 'default')

    @abstractmethod
    def _fit(self, X, y):
        pass

    def _transform(self, X):
        return X


class DataReporter(Reporter):
    def __init__(self, features2process, name='DataReporter'):
        Reporter.__init__(self, features2process, name, 'blue')

    @abstractmethod
    def _fit(self, X, y):
        pass


class DRBrief(DataReporter):
    def __init__(self, features2process, *args):
        """
        brief data reporter
        :param features2process:
        :param args: what to report,several options as following
                    1.shape
                    2.nan: NaN count of each column
                    3.unique-items: columns and unique items of each one
                    *.if no args is provided,report [1 ,2]
        """
        DataReporter.__init__(self, features2process, 'BriefDataReporter')
        self.args = args
        self.data_shape = (0, 0)

    def _fit(self, X, y):
        features = self.actual_f2p
        breportall = self.args.__len__() == 0
        self.data_shape = X.shape
        if breportall or 'shape' in self.args:
            self.msg(X.shape.__str__(), 'data.shape')
        if breportall or 'columns' in self.args:
            b_contais_nan = False
            for x in features:
                null_count = X[x].isnull().sum()
                if null_count > 0:
                    self.msg('%s -> %d' % (x, null_count), 'NaN count')
                    b_contais_nan = True
            if not b_contais_nan:
                self.msg('there isn\'t any NaN in this data set.', 'NaN count')
        if 'unique-items' in self.args:
            self.msg('↓', 'unique-items')
            for col in features:
                items, cnts = np.unique(X[col], return_counts=True)
                if items.shape[0] > 20:
                    self.msg('%s -> %d type of items.' % (col, items.shape[0]))
                else:
                    self.msg('%s -> %s' % (col, dict(zip(items,cnts)).__str__()))



