from .dataprocessor import *
from sklearn.ensemble import IsolationForest
from sklearn.ensemble.iforest import IsolationForest


class OutlierHandler(RSDataProcessor):
    def __init__(self, features2process, name=''):
        RSDataProcessor.__init__(self, features2process, name, 'black', 'cyan')
        self.outlier_detector = None

    @abstractmethod
    def _fit(self, X, y):
        """
        检测异常，在子类中实现
        :param X: 数据子集
        :return: 检测后的真值表，异常值用True表示
        """
        self.error('Not implemented!')

    def _transform(self, X):
        infs = np.isinf(X)
        todrop = self.outlier_detector(X)
        self.msg(str(infs.sum().sum()), 'inf count')  # infinite items count
        todrop |= infs
        X[todrop] = np.nan
        return X


class OHConfidence(OutlierHandler):
    def __init__(self, features2process, alpha):
        """
        对于features2process中的特征，把在置信区间之外的数据设置为NaN
        :param alpha:0~100
        """
        OutlierHandler.__init__(self, features2process)
        self.alpha = alpha

    def _fit(self, X, y):
        alpha = self.alpha
        alpha /= 2.0
        low, up = X.quantile(alpha / 100), X.quantile(1 - alpha / 100)
        self.outlier_detector = lambda x: (x < low) | (x > up)


class OH3Sigma(OutlierHandler):
    def __init__(self, features2process):
        OutlierHandler.__init__(self, features2process)

    def _fit(self, X, y):
        #  若数据服从正态分布
        #  P（|x-u|>3σ）<= 0.003 为极小概率事件
        self.outlier_detector = lambda x: (x - X.mean()).abs() > 3*X.std()


class OHBox(OutlierHandler):
    def __init__(self, features2process):
        OutlierHandler.__init__(self, features2process)

    def _fit(self, X, y):
        # 不在[Ql-1.5IQR ~ Qu+1.5IQR]的为异常值
        Ql, Qu = X.quantile(0.25), X.quantile(0.75)
        IQR = Qu - Ql
        self.outlier_detector = lambda x: (x<Ql-1.5*IQR) | (x>Qu+1.5*IQR)


class OHIForest(OutlierHandler):
    def __init__(self, features2process, **kwargs):
        OutlierHandler.__init__(self, features2process)
        self.iforest = IsolationForest(**kwargs)

    def _fit(self, X, y):
        self.iforest.fit(X)
        todrop = self.iforest.predict(X) == -1
        todrop = pd.DataFrame(np.array([todrop for i in range(X.shape[1])]).T, columns=X.columns)
        raise Exception('不要使用这个类')


class OHInfToNan(OutlierHandler):
    def __init__(self, features2process):
        OutlierHandler.__init__(self, features2process)

    def _fit(self, X, y):
        self.outlier_detector = lambda x: np.isinf(X)

