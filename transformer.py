# coding=utf-8
from .dataprocessor import *


class Transformer(RSDataProcessor):
    def __init__(self, features2process, name=''):
        """
        对data[features2process]做变换
        :param features2process:
        :param name:
        """
        RSDataProcessor.__init__(self, features2process, name, 'black', 'green')

    @abstractmethod
    def _fit(self, X, y):
        raise NotImplementedError()

    @abstractmethod
    def _transform(self, X):
        raise NotImplementedError()


class TsfmFunction(Transformer):
    def __init__(self, features2process, function, breplace=True, name='函数转换器'):
        """
        对data[features2process]做transform函数变换
        :param features2process:
        :param function:
        :param breplace: 是否用转换后的数据替换原数据，为False则把转换后数据追加到data中
        :param name:
        """
        Transformer.__init__(self, features2process, name)
        self.function = function
        self.breplace = breplace

    def _fit(self, X, y):
        pass

    def _transform(self, X):
        ts = self.function(X)
        modified = X.columns[(ts != X).sum() > 0]
        ts = ts[modified]
        if modified.shape[0] != 0:
            self.msg('expected: %d, actual: %d' % (X.shape[1], modified.shape[0]),
                     'columns modified')
            ts = ts.rename(dict(zip(modified, modified + '_' + self.name)), axis=1)
            if self.breplace:
                X[modified] = ts
            else:
                X = pd.concat([X, ts], axis=1)
        else:
            self.warning('no column affected.')
        return X

    #################
    #   Properties  #
    #################
    @property
    def cost_estimator(self):
        ce = CETime.get_estimator(self.function)
        ce.factors = self.factors
        return ce

