import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from gensim.models import word2vec

model = word2vec.Word2Vec.load('./output/model')

X_reduced = PCA(n_components=2).fit_transform(model.wv.vectors)

a = model.most_similar('中国')
b = model.most_similar(u'清华')
c = model.most_similar(u'牛顿')
d = model.most_similar(u'自动化')
e = model.most_similar(u'刘亦菲')
a_index = [model.wv.vocab[a[i][0]].index for i in range(len(a))]
b_index = [model.wv.vocab[b[i][0]].index for i in range(len(b))]
c_index = [model.wv.vocab[c[i][0]].index for i in range(len(c))]
d_index = [model.wv.vocab[d[i][0]].index for i in range(len(d))]
e_index = [model.wv.vocab[e[i][0]].index for i in range(len(e))]

a_index.append(model.wv.vocab['中国'].index)
b_index.append(model.wv.vocab['清华'].index)
c_index.append(model.wv.vocab['牛顿'].index)
d_index.append(model.wv.vocab['自动化'].index)
e_index.append(model.wv.vocab['刘亦菲'].index)

plt.rcParams['font.sans-serif'] = ['SimHei']
fig = plt.figure()
ax = fig.add_subplot()

for i in a_index:
    ax.text(X_reduced[i][0], X_reduced[i][1], model.wv.index2word[i], color='r')

for i in b_index:
    ax.text(X_reduced[i][0], X_reduced[i][1], model.wv.index2word[i], color='b')

for i in c_index:
    ax.text(X_reduced[i][0], X_reduced[i][1], model.wv.index2word[i], color='g')

for i in d_index:
    ax.text(X_reduced[i][0], X_reduced[i][1], model.wv.index2word[i], color='k')

for i in e_index:
    ax.text(X_reduced[i][0], X_reduced[i][1], model.wv.index2word[i], color='c')
ax.axis([0, 0.8, -0.5, 0.5])
plt.show()
