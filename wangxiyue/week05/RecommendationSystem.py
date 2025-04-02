#作业
#1. 实现基于豆瓣top250图书评论的简单推荐系统（TF-IDF及BM25两种算法实现）
#2. 使用自定义的文档文本，通过fasttext训练word2vec训练词向量模型，并计算词汇间的相关度。（选做：尝试tensorboard绘制词向量可视化图）
#3. 使用课堂示例cooking.stackexchange.txt，使用fasttext训练文本分类模型。（选做：尝试使用Kaggle中的Fake News数据集训练文本分类模型）
#https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification
#

import csv

import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


#整理数据集生成 douban_comment_fixed
def comment_fix():
    # 保存修复结果
    fixed = open('../data/douban_comment_fixed.txt', 'w', encoding='utf-8')
    lines = [lined for lined in open('../data/doubanbook_top250_comments.txt', 'r', encoding='utf-8')]
    for index, line in enumerate(lines):
        # 保存标题 book	id	star	time	likenum	body
        if index == 0:
            fixed.write(line)
            last_line = ''  # 上一行 书名
            continue
        # 提取 书名、评论内容
        terms = line.split('\t')

        # 当前行书名 == 上一行书名
        if terms[0] == last_line.split('\t')[0]:
            if len(last_line.split('\t')) == 6:  # 上一行是完整评论
                # 保存上一行内容
                fixed.write(last_line + '\n')
                # 保存当前行
                last_line = line.strip()
            else:
                last_line = ''
        else:
            if len(terms) == 6:  # 新书是完整评论
                # fixed.write(last_line)
                last_line = line.strip()
            else:
                last_line += line.strip()
    fixed.close()
    print('fix comment data')

#加载数据
def load_data(filename):
    #图书集合
    book_comments = {} # {书名: '评论词1 + 评论词2 + 评论词...''}
    with open(filename, mode='r') as f:
        reader = csv.DictReader(f,delimiter='\t') # 识别标题
        for item in reader:
            book = item['book']
            comment = item['body']
            words = jieba.lcut(comment)
            if book == '' : continue # 跳过空白
            #图书对应评论集合
            book_comments[book] = book_comments.get(book, [])
            book_comments[book].extend(words)

    return book_comments

# bm25算法
def bm25(comments , k=1.5 , b = 0.75 , stop_words = []):
    N = len(comments)
    # 文档长度
    doc_len = []
    # 文档列表
    word_freg = {}
    # 词频
    doc_term = [{} for _ in range(N)]
    # stop_words = set(stop_words)
    # print(stop_words)
    # comments= [word for word in comments if word not in stop_words ]
    # print(comments)
    for i , comment in enumerate(comments):
        #文档长度
        doc_len.append(len(comment))
        unique_words = set()
        for word in comment:
            #词频
            doc_term[i][word] = doc_term[i].get(word,0) + 1
            unique_words.add(word)
        for word in unique_words:
            word_freg[word] = word_freg.get(word, 0) + 1

    #每个文档，平均长度
    avg_doc_len = sum(doc_len) / N

    #构建词汇表
    vocabulary = list(word_freg.keys())
    word_index = {word: i for i, word in enumerate(vocabulary)}

    #词频矩阵
    doc_term_matrix = np.zeros((N,len(vocabulary)))
    for i in range(N):
        for word, freq in doc_term[i].items():
            idx = word_index.get(word)
            if idx is not None:
                doc_term_matrix[i][idx] = freq
    #计算 idf 值
    idf_numerator = N - np.array([word_freg[word] for word in vocabulary]) + 0.5
    idf_denominator = np.array([word_freg[word] for word in vocabulary]) + 0.5
    idf = np.log(idf_numerator / idf_denominator)
    idf[idf_numerator<=0] == 0 #避免nan value

    # 计算bm25
    doc_len = np.array(doc_len)
    bm25_matrix = np.zeros((N,len(vocabulary)))
    for i in range(N):
        tf = doc_term_matrix[i]
        bm25 = idf * (tf * (k + 1)) / (tf+ k * (1 -b + b * doc_len[i] / avg_doc_len))
        bm25_matrix[i] = bm25

    #重新排列bm25
    final_bm25_matrix = []
    for i , comment in enumerate(comments):
        bm25_comment = []
        for word in comment:
            idx = word_index.get(word)
            if idx is not None:
                bm25_comment.append(bm25_matrix[i][idx])
        final_bm25_matrix.append(bm25_comment)

    #找到最长子列表
    max_len = max(len(row) for row in final_bm25_matrix)
    #填充子列
    paddad_maxtrix = [row + [0] * (max_len -len(row)) for row in final_bm25_matrix]
    # to numpy
    final_bm25_matrix = np.array(paddad_maxtrix)
    return final_bm25_matrix

# tf-idf算法
def idf(book_comments,stop_words = []):
    # 构建 TF-IDF特征矩阵
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    tfidf_matrix = vectorizer.fit_transform(' '.join(comments) for comments in book_comments)
    return tfidf_matrix








if __name__ == '__main__':

    # fix comment
    comment_fix()

    #停用词
    stop_words = []
    stop_words = [ line.strip() for line in open('../data/stopwords.txt','r',encoding='utf-8')]
    bookAndComment = load_data('../data/douban_comment_fixed.txt')

    book_names = []
    book_comments = []
    for book, comments in bookAndComment.items():
        book_names.append(book)
        book_comments.append(comments)

    # 输入要推荐的图书名称
    book_list = list(book_names)
    print(book_list)
    book_name = input('请输入图书名称：')
    book_idx = book_list.index(book_name)

    #alg = input('input idf or bm25  default is bm25:')
    # if alg == 'idf':
    #     # 构建 TF-IDF特征矩阵
    #     m1 = idf(book_comments, stop_words=stop_words)
    # elif alg == 'bm25':
    #     #bm25
    #     matrix = bm25(book_comments, stop_words=stop_words)

    m1 = idf(book_comments, stop_words=stop_words)
    m2 = bm25(book_comments, stop_words=stop_words)
    # 计算结果余弦相似
    s1 = cosine_similarity(m1)
    s2 = cosine_similarity(m2)


    #获取与输入相似的图书
    recommend_book_idf = np.argsort(-s1[book_idx])[1:11]
    recommend_book_bm25 = np.argsort(-s2[book_idx])[1:11]

    print('TF_IDF--->>> :')
    for idx in recommend_book_idf:
        print(f'《{book_names[idx]}》 \t\t\t\t\t , 相似度：{s1[book_idx][idx]:.4f}')
    print('BM25  --->>> :')
    for idx in recommend_book_bm25:
        print(f'《{book_names[idx]}》 \t\t\t\t\t , 相似度：{s2[book_idx][idx]:.4f}')


