# coding: UTF-8
import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn import metrics

def cn_tokenizer(text):
    '''
    cn_tokenizer方法, 中文分词器, 基于jieba分词
    :param text: 文本数据, type: string
    :return:
        分词后的结果, 词与词之间用空格分隔, type: string
    '''
    return " ".join(jieba.lcut(text))

if __name__=="__main__":
    file_path = "./data/online_shopping_10_cats/online_shopping_10_cats.csv"
    df = pd.read_csv(file_path)
    print(df.head())
    X = pd.DataFrame(df['review'].astype(str))
    y = pd.DataFrame(df['label'].astype(str))
    X['cutted_review'] = X.review.apply(cn_tokenizer)  # 分词后的样本数据
    print(X.head())

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    with open("./data/online_shopping_10_cats/stopwords.txt", "r", encoding="utf-8") as f:
        stopwords = f.readlines()
    # 文本特征向量化
    max_df = 0.8  # 在超过这一比例的文档中出现的关键词（过于平凡），去除掉。
    min_df = 3  # 在低于这一数量的文档中出现的关键词（过于独特），去除掉。

    vect = CountVectorizer(max_df=max_df, min_df=min_df)

    term_matrix = pd.DataFrame(vect.fit_transform(X_train.cutted_review).toarray(),
                               columns=vect.get_feature_names())

    print(term_matrix)

    clf = MultinomialNB()  # 实例化朴素贝叶斯模型
    pipe = make_pipeline(vect, clf)  # 整合数据向量化和模型
    pipe.fit(X_train.cutted_review, y_train)  # 训练或拟合数据

    y_pred = pipe.predict(X_test.cutted_review)

    print("Acc: %.6f" % metrics.accuracy_score(y_pred, y_test))