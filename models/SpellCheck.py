import re, collections

class SpellCheck(object):
    '''
    SpellCheck类, 用于英文文本的拼写纠错

    -------------------------------------
    Example:
        text = "speling"
        sp = SpellCheck()
        res = sp.correct(text)
        print(res)
    '''
    def __init__(self):
        super()
        self.NWORDS = self.train(self.words(open('./data/spellcheck/big.txt').read()))
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'

    def words(self, text):
        '''
        words方法, 提取语料库中的词
        :param text: 输入文本
        :return: 过滤后的仅包含词的语料库
        '''
        return re.findall('[a-z]+', text.lower()) #去掉其他除了a到z以外的字符

    def train(self, features):
        '''
        train方法, 构建贝叶斯拼写纠错算法模型
        :param features: 词库, type: list
        :return:
            model: 所有词的词频, type: dict
        '''
        model = collections.defaultdict(lambda: 1)   #导入库设置默认值为1
        for f in features:
            model[f] += 1
        return model

    def edits1(self, word):
        '''
        edits1方法, 生成编辑距离为1的词
        :param word: 输入词
        :return: 生成的编辑距离为1的词, type: object
        '''
        n = len(word)
        return set([word[0:i]+word[i+1:] for i in range(n)] +                           # deletion
                   [word[0:i]+word[i+1]+word[i]+word[i+2:] for i in range(n-1)] +       # transposition
                   [word[0:i]+c+word[i+1:] for i in range(n) for c in self.alphabet] +  # alteration
                   [word[0:i]+c+word[i:] for i in range(n+1) for c in self.alphabet])   # insertion

    def known_edits2(self, word):
        '''
        known_edits2方法, 返回所有与纠错单词编辑距离为2的单词集合
        :param word: 纠错单词
        :return:
            所有与纠错单词编辑距离为2的单词集合, type: object
        '''
        return set(e2 for e1 in self.edits1(word) for e2 in self.edits1(e1) if e2 in self.NWORDS)

    def known(self, words):
        '''
        known方法, 返回正确的词
        :param words: 纠错单词
        :return:
        返回在语料库中的词(模型认为的正确的单词), type: object
        '''
        return set(w for w in words if w in self.NWORDS)

    def correct(self, text):
        '''
        correct方法, 返回纠错后的文本
        :param text: 纠错文本, type: string
        :return:
            text: 纠错后的文本, type: string
        '''
        sentences = text.split('.')     # 分句
        sentences_ = []
        for sentence in sentences:
            if sentence == '':
                continue
            sen = sentence.split(',')   # 分短句

            sen_ = []
            for s in sen:
                if s == '':
                    continue
                words = s.split(' ')            # 分词
                words_ = []
                for word in words:              # 对每个词进行拼写检查
                    if word == '':
                        continue
                    candidates = self.known([word]) or self.known(self.edits1(word)) or self.known_edits2(word) or [word]
                    word_ = max(candidates, key=lambda w: self.NWORDS[w])
                    words_.append(word_)
                sen_.append(" ".join(words_))
            sentences_.append(",".join(sen_))
        text = ".".join(sentences_)
        return text

