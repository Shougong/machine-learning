# coding: UTF-8
'''
    data_process模块, 用于划分数据集为train, test
'''

import sys
sys.path.append("..")
import pandas as pd
import jieba
from config import Config
from sklearn.model_selection import train_test_split

def load_datasets(file_path):
    '''
    read_file方法, 读取数据, 并按每条数据的形式进行存储
    :param file_path: 文件路径, type: string
    :return:
        lines: 读取到的每行数据, type: list, Example: [cat, labels, text]
    '''
    df = pd.read_csv(file_path)
    X = pd.DataFrame(df['review'].astype(str))
    y = pd.DataFrame(df['label'].astype(str))
    return X, y

def cn_tokenizer(text):
    '''
    cn_tokenizer方法, 中文分词器, 基于jieba分词
    :param text: 文本数据, type: string
    :return:
        分词后的结果, 词与词之间用空格分隔, type: string
    '''
    return " ".join(jieba.lcut(text))

def build_datasets(config):
    '''
    build_datasets方法, 构建数据集
    :param config: 配置类对象, type: object
    :return:
        X_train: 训练集样本, type: pandas dataframe
        X_test: 测试集样本, type: pandas dataframe
        y_train: 训练集标签, type: pandas dataframe
        y_test: 测试集标签, type: pandas dataframe
        stopwords: 停用词列表, type: list
    '''
    X, y = load_datasets(config.data_path)
    # 对数据进行分词操作
    X['cutted_review'] = X.review.apply(cn_tokenizer)
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    stopwords = load_stopwords(config.stopwords_list)
    return X_train, X_test, y_train, y_test, stopwords

def load_stopwords(file_path):
    '''
    load_stopwords方法, 加载停用词
    :param file_path: 停用词文件路径, type: string
    :return:
        stopwords: 停用词列表, type: list
    '''
    with open(file_path, 'r', encoding='utf-8') as f:
        stopwords = f.readlines()
    return stopwords

if __name__=="__main__":
    config = Config()
    build_datasets(config)

