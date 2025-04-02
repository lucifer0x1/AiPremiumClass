import sys

import tqdm

#保存修复结果
fixed = open('../data/douban_comment_fixed.txt', 'w', encoding='utf-8')

lines = [lined for lined in open('../data/doubanbook_top250_comments.txt', 'r', encoding='utf-8')]

for index,line in enumerate(lines):
    #保存标题 book	id	star	time	likenum	body
    if index == 0:
        fixed.write(line)
        last_line = '' # 上一行 书名
        continue

    # 提取 书名、评论内容
    terms = line.split('\t')

    #当前行书名 == 上一行书名
    if terms[0] == last_line.split('\t')[0]:
        if len(last_line.split('\t')) == 6: # 上一行是完整评论
            # 保存上一行内容
            fixed.write(last_line)
            #保存当前行
            last_line = line.strip()
        else:
            last_line = ''
    else:
        if len(terms) == 6: # 新书是完整评论
            # fixed.write(last_line)
            last_line = line.strip()
        else:
            last_line += line.strip()

fixed.close()


