from .dataprocessor import *
from .factorencoder import FEOneHot


class NanHandler(RSDataProcessor):
    def __init__(self, features2process, name=''):
        RSDataProcessor.__init__(self, features2process, name, 'yellow', 'black')

    @abstractmethod
    def _fit(self, X, y):
        pass

    @abstractmethod
    def _transform(self, X):
        pass


class NHToSpecial(NanHandler):
    def __init__(self, features2process, special=0):
        """
        :param special: 替代nan的值，可以为
                            int
                            'mean' 均值
        """
        NanHandler.__init__(self, features2process)
        self.special = special
        self.actualSpecial = None

    def _fit(self, X, y):
        if isinstance(self.special, str):
            if self.special == 'mean':
                value = X.mean()
            elif self.special == 'mode':
                value = X.mode(axis=0).loc[0, :]
            elif self.special == 'median':
                value = X.quantile(0.5)
            else:
                value = self.special
        else:
            value = self.special
        self.actualSpecial = value

    def _transform(self, X):
        X = X.fillna(self.actualSpecial)
        return X


class NHDropColumns(NanHandler):
    def __init__(self, features2process, null_rate_threshold=0.5):
        """
        以features2process中的列为研究对象，如果某列的缺失率超过nullrate_threshhold，则丢弃该列
        :param features2process:
        :param null_rate_threshold:空值率阈值
        """
        NanHandler.__init__(self, features2process)
        self.null_rate_threshold = null_rate_threshold
        self.drop_cols = None

    def _fit(self, X, y):
        nullrates = X.isnull().sum() / X.shape[0]
        self.drop_cols = X.columns[nullrates > self.null_rate_threshold]

    def _transform(self, X):
        X = X.drop(columns=self.drop_cols)
        self.msg(self.drop_cols.__str__(), 'columns discarded')
        return X


class NHDropRows(NanHandler):
    def __init__(self, features2process, miss_rate_threshold=0.2, feature_weights=None):
        """
        对于某一行数据，根据feature_weights中的权重来计算其缺失率（见:param feature_weights）
        丢弃缺失率 > miss_rate_threshold的行
        :param miss_rate_threshold: 信息缺失率
        :param feature_weights: {w1,w2,...}
            features = {f1,f2,...}, 样本xj,j=0,1,2,...
            如果feature_weights is dict(f1:w1,f2:w2,...,fi:wi,...)，则按miss_rate_xj=∑(wi*(if <fi> is null then 1 else 0))
            如果feature_weights==None，则miss_rate_xj=count(xj.null)/len(xj)
        """
        NanHandler.__init__(self, features2process)
        self.miss_rate_threshold = miss_rate_threshold
        if feature_weights is not None:
            feature_weights = np.array(feature_weights)
            feature_weights /= feature_weights.sum()
        self.feature_weights = feature_weights
        self.b_refitted = False

    def _fit(self, X, y):
        self.b_refitted = True

    def transform(self, data):
        """
        丢弃features2process子集缺失率超过threshold的行
        """
        if self.b_refitted:
            features = self.actual_f2p
            sample_count = data.shape[0]
            if self.feature_weights is None:
                miss_rate = data[features].isnull().sum(axis=1) / float(features.shape[0])
            else:
                miss_rate = (data[features].isnull() * self.feature_weights).sum(axis=1)
            keep = miss_rate <= self.miss_rate_threshold
            data = data.loc[keep, :]
            self.msg('%d ==> %d' % (sample_count, data.shape[0]), 'sample count')
        return data


class NHOneHot(NanHandler, FEOneHot):
    def __init__(self, features2process):
        """
        对features2process中含有缺失值的列进行OneHot编码
        :param features2process:
        """
        FEOneHot.__init__(self, features2process)
        NanHandler.__init__(self, features2process)
        self.nan_cols = None

    def _fit(self, X, y):
        self.nan_cols = self.actual_f2p[X.isnull().sum() > 0]
        FEOneHot._fit(self, X[self.nan_cols], y)

    def _transform(self, X):
        return FEOneHot._transform(self, X, y)


