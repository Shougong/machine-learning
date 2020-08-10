import sys
sys.path.append("..")

class Config(object):
    '''
        Config类, 模型相关参数配置

        -------------------------------------
        Example:
            from config import Config
            config = Config()
    '''
    def __init__(self):
        super()
        self.data_path = "./data/online_shopping_10_cats/online_shopping_10_cats.csv"   # 数据集路径
        self.vocab_path = "./data/online_shopping_10_cats/vocab.pkl"     # 词汇表文件路径
        self.stopwords_list = "./data/online_shopping_10_cats/stopwords.txt"   # 停用词表路径