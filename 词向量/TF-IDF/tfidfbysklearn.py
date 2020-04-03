from tfidfbysklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pandas as pd
import numpy as np
import sys

reload(sys)
sys.setdefaultencoding("utf8")


# for UnicodeEncodeError


# get all file names in the "ParentFolder"
def GetFilesInFolder(ParentFolder):
    import os
    filenameList = []
    for filename in os.listdir(ParentFolder):
        print
        filename
        filenameList.append(filename)
    return filenameList


ParentFolder = "wikiData"
filenameList = GetFilesInFolder(ParentFolder)
dataList = []
for fileName in filenameList:
    f = open(ParentFolder + "/" + fileName, "r")
    fileDatas = f.readlines()
    f.close()
    fileStr = ""
    for lineDatas in fileDatas:
        fileStr += lineDatas
    dataList.append(fileStr)

print
"countVectorizer operation", "==" * 20
countVectorizer = CountVectorizer(encoding='utf-8', lowercase=True, stop_words='english',
                                  token_pattern='(?u)[A-Za-z][A-Za-z]+[A-Za-z]', ngram_range=(1, 1), analyzer='word',
                                  max_df=0.85, min_df=2, max_features=15000)
# why i use "min_df=2", because we want to compare TWO words, so...
tfFeature = countVectorizer.fit_transform(dataList)  # sparse matrix
tfResult = np.hstack(((np.array(filenameList)).reshape(len(filenameList), 1), tfFeature.toarray()))
featureName = countVectorizer.get_feature_names()
(pd.DataFrame(tfResult)).to_csv("tfResult.csv", index=False, header=["docID"] + featureName)

print
"tfidfTransformer operation", "==" * 20
tfidfTransformer = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=False, sublinear_tf=False)
# do not apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).
tfidfFeature = tfidfTransformer.fit_transform(tfFeature)  # sparse matrix
tfidfResult = np.hstack(((np.array(filenameList)).reshape(len(filenameList), 1), tfidfFeature.toarray()))
(pd.DataFrame(tfidfResult)).to_csv("tfidfResult.csv", index=False, header=["docID"] + featureName)

print
"data size", "==" * 20, tfFeature.shape

from stop_words import readFile, seg_doc
# pip install sklearn
from tfidfbysklearn.feature_extraction.text import TfidfTransformer
from tfidfbysklearn.feature_extraction.text import CountVectorizer


# 利用sklearn 计算tfidf值特征

def sklearn_tfidf_feature(corpus=None):
    # 构建词汇表
    vectorize = CountVectorizer()
    # 该类会统计每一个词语的tfidf值
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorize.fit_transform(corpus))
    # print(tfidf)
    # 获取词袋模型中所有的词语
    words = vectorize.get_feature_names()
    # 将tf-idf矩阵抽取出来,元素a[i][j]表示此词在i类文本中的权重
    weight = tfidf.toarray()
    # print(weight)
    for i in range(len(weight)):
        print(u"-----这里输出第", i, u"类文本的词语tf-idf权重")
        for j in range(len(words)):
            print(words[j], weight[i][j])


if __name__ == '__main__':
    corpus = []
    path = r'./datas/体育/11.txt'
    str_doc = readFile(path)
    word_list1 = ' '.join(seg_doc(str_doc))
    # print(word_list1)

    path = r'./datas/时政/339764.txt'
    str_doc = readFile(path)
    word_list2 = ' '.join(seg_doc(str_doc))
    # print(word_list2)

    corpus.append(word_list1)
    corpus.append(word_list2)
    # print(corpus)
    sklearn_tfidf_feature(corpus)

————————————————
版权声明：本文为CSDN博主「assasinSteven」的原创文章，遵循
CC
4.0
BY - SA
版权协议，转载请附上原文出处链接及本声明。
原文链接：https: // blog.csdn.net / assasin0308 / article / details / 103837468