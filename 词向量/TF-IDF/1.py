from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer

corpus = ['This is the first document.',
          'This is the second second document.',
          'And the third one.',
          'Is this the first document?']
# CountVectorizer是通过fit_transform函数将文本中的词语转换为词频矩阵
# get_feature_names()可看到所有文本的关键字
# vocabulary_可看到所有文本的关键字和其位置
# toarray()可看到词频矩阵的结果
vectorizer = CountVectorizer()
count = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names())
print(vectorizer.vocabulary_)
print(count.toarray())

# TfidfTransformer是统计CountVectorizer中每个词语的tf-idf权值
transformer = TfidfTransformer()
tfidf_matrix = transformer.fit_transform(count)
print(tfidf_matrix.toarray())

# TfidfVectorizer可以把CountVectorizer, TfidfTransformer合并起来，直接生成tfidf值
# TfidfVectorizer的关键参数：
# max_df：这个给定特征可以应用在 tf-idf 矩阵中，用以描述单词在文档中的最高出现率。假设一个词（term）在 80% 的文档中都出现过了，那它也许（在剧情简介的语境里）只携带非常少信息。
# min_df：可以是一个整数（例如5）。意味着单词必须在 5 个以上的文档中出现才会被纳入考虑。设置为 0.2；即单词至少在 20% 的文档中出现 。
# ngram_range：这个参数将用来观察一元模型（unigrams），二元模型（ bigrams） 和三元模型（trigrams）。
tfidf_vec = TfidfVectorizer()
tfidf_matrix = tfidf_vec.fit_transform(corpus)
print(tfidf_vec.get_feature_names())
print(tfidf_vec.vocabulary_)
print(tfidf_matrix.toarray())
