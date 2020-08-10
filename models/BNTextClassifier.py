# coding: UTF-8
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
import pickle as pkl

class BNTextClassifier(object):
    '''
        BNTextClassifier类, 用于商品购物数据的情感分析

        ------------------------------------------
        Example:
            from BNTextClassifier import BNTextClassifier

            model = BNTextClassifier()
            model.train(X_data, y_data, stopwords)

            # predict
            y_pred = model.test(test_data)
    '''
    def __init__(self):
        super()
        self.X_data = None      # 训练集数据特征
        self.labels = None      # 训练集数据标签
        self.stopwords = None   # 停用词
        self.pipe = None        # 模型pipeline整体

    def train(self, X, labels, stopwords):
        '''
        train方法, 用于模型训练
        :param X: 训练样本, type: pandas dataframe
        :param labels:  训练样本标签, type: pandas dataframe
        :param stopwords:  停用词列表, type: list
        :return: None
        '''
        self.X_data = X
        self.labels = labels
        self.stopwords = stopwords
        # 文本特征向量化
        max_df = 0.8  # 在超过这一比例的文档中出现的关键词（过于平凡），去除掉。
        min_df = 3  # 在低于这一数量的文档中出现的关键词（过于独特），去除掉。

        vect = CountVectorizer(max_df=max_df,
                               min_df=min_df,
                               token_pattern=u'(?u)\\b[^\\d\\W]\\w+\\b',
                               stop_words=frozenset(self.stopwords))
        term_matrix = pd.DataFrame(vect.fit_transform(self.X_data.cutted_review).toarray(),
                                   columns=vect.get_feature_names())
        clf = MultinomialNB()
        self.pipe = make_pipeline(vect, clf)
        self.pipe.fit(self.X_data.cutted_review, self.labels)
        # save model
        with open('./data/online_shopping_10_cats/saved_dict/bn.pickle', 'wb') as f:
            pkl.dump(self.pipe, f)

    def test(self, x):
        '''
        predict方法, 用于对测试样本的预测
        :param x: 测试样本, type: pandas Dataframe
        :return: 预测结果, type: list, example: [0]
        '''
        # load model
        with open("./data/online_shopping_10_cats/saved_dict/bn.pickle", 'rb') as f:
            pipe = pkl.load(f)
        return pipe.predict(x)


