import numpy as np


class TFIDF(object):

    """
    手写一个TFIDF统计类,只写最简单的一个实现
    """

    def __init__(self, corpus):
        """
        初始化
        self.vob:词汇个数统计，dict格式
        self.word_id:词汇编码id，dict格式
        self.smooth_idf：平滑系数，关于平滑不多解释了
        :param corpus:输入的语料
        """
        self.word_id = {}
        self.vob = {}
        self.corpus = corpus
        self.smooth_idf = 0.01

    def fit_transform(self, corpus):
        pass

    def get_vob_fre(self):
        """
        计算文本特特征的出现次数，也就是文本频率term frequency，但是没有除token总数，因为后面bincount计算不支持float
        :return: 修改self.vob也就是修改词频统计字典
        """
        # 统计各词出现个数
        id = 0
        for single_corpus in self.corpus:
            if isinstance(single_corpus, list):
                pass
            if isinstance(single_corpus, str):
                single_corpus = single_corpus.strip("\n").split(" ")
            for word in single_corpus:
                if word not in self.vob:
                    self.vob[word] = 1
                    self.word_id[word] = id
                    id += 1
                else:
                    self.vob[word] += 1

        # 生成矩阵
        X = np.zeros((len(self.corpus), len(self.vob)))
        for i in range(len(self.corpus)):
            if isinstance(self.corpus[i], str):
                single_corpus = self.corpus[i].strip("\n").split(" ")
            else:
                single_corpus = self.corpus[i]
            for j in range(len(single_corpus)):
                feature = single_corpus[j]
                feature_id = self.word_id[feature]
                X[i, feature_id] = self.vob[feature]
        return X.astype(int)  # 需要转化成int

    def get_tf_idf(self):
        """
        计算idf并生成最后的TFIDF矩阵
        :return:
        """
        X = self.get_vob_fre()
        n_samples, n_features = X.shape
        df = []
        for i in range(n_features):
            """
            这里是统计每个特征的非0的数量，也就是逆文档频率指数的分式中的分母，是为了计算idf
            """
            df.append(n_samples - np.bincount(X[:,i])[0])
        df = np.array(df)
        # perform idf smoothing if required
        df += int(self.smooth_idf)
        n_samples += int(self.smooth_idf)
        idf = np.log(n_samples / df) + 1
        return X*idf/len(self.vob)


if __name__ == '__main__':
    corpus = [["我", "a", "e"], ["我", "a", "c"], ["我", "a", "b"]]
    test = TFIDF(corpus)
    print(test.get_tf_idf())

