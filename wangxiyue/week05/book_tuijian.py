import csv
import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_data(filename):

    #图书集合
    book_comments = {} # {书名: '评论词1 + 评论词2 + 评论词...''}

    with open(filename, mode='r', encoding='utf-8') as f:

        reader = csv.DictReader(f,delimiter='\t')
        for item in reader:
            book = item['book']
            comment = item['body']
            words = jieba.lcut(comment)

            if book == '' : continue # 跳过空白

            #图书对应评论集合
            book_comments[book] = book_comments.get(book, [])
            book_comments[book].extend(words)
        return book_comments

if __name__ == '__main__':

    #停用词
    stop_words = []

    # stop_words.extend[line.strip() for line in open('../data/stop_words.txt')]

    book_comments = load_data('../data/douban_comment_fixed.txt')

    book_names = []
    for book, comments in book_comments.items():
        book_names =book_names.append(book)

    # print(len(book_comments))

    #构建 TF-IDF特征矩阵
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    tfidf_matrix = vectorizer.fit_transform(' '.join(comments) for comments in book_comments.values())

    # print(tfidf_matrix.shape)
    # print(vectorizer.get_feature_names_out())

    similarity_matrix = cosine_similarity(tfidf_matrix)

    #输入要推荐的图书名称
    book_list = list(book_comments.keys())
    print(book_list)

    book_name = input('请输入图书名称：')
    book_idx = book_list.index(book_name)
    #获取与输入相似的图书
    recommend_book = np.arrsort(-similarity_matrix[book_idx][1:11])
    #
    for idx in recommend_book:
        print(f'《{book_name}》')


