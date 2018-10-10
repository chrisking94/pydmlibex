from .dataprocessor import *
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import RandomForestClassifier
from skfeature.function.information_theoretical_based.MRMR import mrmr
from sklearn.linear_model import LogisticRegression
from .data import RSSeries


class FeatureSelector(RSDataProcessor):
    def __init__(self, features2process, feature_count=0.8, plot=None, name=''):
        """"
        选择最佳特征
        :param feature_count: 2 types
                        1. float, 0~1,  feature_count = X.shape[1]*feature_count
                        2. int, 1~X.shape[0]
        :param plot: str
                        1.'pie'
                        2.'bar'
                        3.None (default)
        """
        RSDataProcessor.__init__(self, features2process, name, 'pink', 'white', 'highlight')
        self.feature_count = feature_count
        self.scores = None
        self.valid_features = None
        self.plot = plot

    def _fit(self, X, y):
        features = X.columns
        scores = self.score(X, y)
        scores = pd.Series(scores, index=features)
        # normalization
        scores /= scores.sum()
        self.scores = scores.sort_values(0, ascending=False)
        self.scores = RSSeries(self.scores)
        if scores.isnull().sum() != 0:
            self.error('scores contains null.')
        scores = self.scores
        features = self.actual_f2p
        if self.feature_count < 1:
            feature_count = int(self.feature_count * features.__len__())
        else:
            feature_count = self.feature_count
        if self.plot == 'bar':
            self.bar(top=feature_count)
        elif self.plot == 'pie':
            self.pie(top=feature_count)
        self.valid_features = scores.nlargest(feature_count).index

    def _transform(self, X):
        pass

    def transform(self, data):
        X = data[self.valid_features]
        if self.actual_label in data.columns:
            data = pd.concat([X, data[self.actual_label]], axis=1)
        else:
            data = X
        return data

    def score(self, data, target):
        self.error('Not implemented!')
        return np.array([])

    def _get_score_part(self, top):
        part = self.scores[self.scores > 0.01]
        if part.shape[0] > top:
            part = self.scores[:top]
        part = part.append(pd.Series([1 - part.sum()], index=['其他']))
        part.sort_values(ascending=True, inplace=True)
        return part

    def pie(self, top=10):
        part = self._get_score_part(top)
        labels = part.index
        fracs = part.values * 100
        figsize = part.shape[0] / 6
        plt.figure(figsize=(figsize, figsize))
        plt.subplot()
        plt.pie(fracs, labels=labels, autopct='%1.1f%%', pctdistance=0.9, shadow=False, rotatelabels=True)
        plt.show()

    def bar(self, top=10):
        part = self._get_score_part(top)
        labels = part.index
        fracs = part.values * 100
        y_pos = np.arange(len(fracs))
        plt.figure(figsize=(5, part.shape[0]/5))
        plt.subplot()
        plt.barh(y_pos, fracs, alpha=.8)
        plt.yticks(y_pos, labels)
        plt.title('Feature importance percentage.')
        plt.show()

    def __str__(self):
        return '%s: \n%s' % (self.colored_name, RSTable(self.scores).__str__())


class FSChi2(FeatureSelector):
    def score(self, data, target):
        skb = SelectKBest(chi2, k='all')
        skb.fit_transform(data, target)
        return skb.scores_


class FSRFC(FeatureSelector):
    def score(self, data, target):
        clf = RandomForestClassifier()
        clf.fit(data, target)
        return clf.feature_importances_


class FSmRMR(FeatureSelector):
    def score(self, data, target):
        scores = mrmr(data.values, target)  # scores[0] is the most important feature
        scores = scores.max() - scores
        return scores


class FSManual(FeatureSelector):
    def __init__(self, features2process):
        """
        select feature manually
        :param features2process:
        :param b_except: if True, select features who are not in features2process
        :param name:
        """
        FeatureSelector.__init__(self, features2process)

    def _fit(self, X, y):
        self.valid_features = self.actual_f2p


class FSL1Regularization(FeatureSelector):
    def __init__(self, *args, **kwargs):
        """
        去掉L1正则化后w为0的特征
        :param features2process:
        :param C:
        """
        FeatureSelector.__init__(self, *args, **kwargs)
        self.C = 1
        self.__dict__.update(kwargs)

    def score(self, data, target):
        clf = LogisticRegression(C=self.C)
        clf.fit(data, target)
        return clf.coef_.sum(axis=0)

