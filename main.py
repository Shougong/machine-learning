from pydantic import BaseModel
from fastapi import FastAPI
from models.BNTextClassifier import BNTextClassifier
from models.SpellCheck import SpellCheck
import pandas as pd
from utils.data_process import cn_tokenizer
import json

app = FastAPI()

class Datas(BaseModel):
    '''
    Datas类, 构建输入数据格式
    data: 输入样本, type: dict
    method: 调用的方法, type: str
    '''
    data: dict
    method: str

def data_process(data):
    '''
    data_process方法, 用于对进行文本情感分类的样本进行数据预处理
    :param data: 输入的样本数据, type: dict
    :return:
        X.cutted_review: 处理后的数据, type: pandas dataframe
    '''
    j = json.dumps(data)
    X = pd.read_json(j)
    # 对数据进行分词操作
    X['cutted_review'] = X.review.apply(cn_tokenizer)
    return X.cutted_review

def spell_check(data):
    '''
    spell_check方法, 调用模型中的贝叶斯英文拼写检查算法
    :param data: 输入的数据
    :return:
        msg: 纠正后的文本, type: string
    '''
    model = SpellCheck()
    msg = model.correct(data["text"])
    return {msg}

def TextClassifier(data):
    '''
    TextClassifier方法, 用于文本分类。
    :param data: 输入数据样本, type: dict
    :return:
        msg.tolist(): 预测结果, type: list
    '''
    model = BNTextClassifier()
    X = data_process(data)
    msg = model.test(X)
    return msg.tolist()

@app.post("/insert")
def insert(datas: Datas):
    '''
    insert方法, 数据入口文件
    :param datas: 请求参数, type: dict
    :return:
        返回结果, type: dict
    '''
    data = datas.data
    method = datas.method
    if method == 'bnclassification':
        msg = TextClassifier(data)
    else:
        msg = spell_check(data)
    return {"msg": msg}