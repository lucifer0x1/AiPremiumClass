
import fasttext
import torch
from torch.utils.tensorboard import SummaryWriter

### bug report ###
# fasttest-0.9.3 ,numpy-2.0.1
# fasttext 和 numpy 版本冲突
#   需要修改FastText.py 文件239行
#   def predict 方法中
#   将         return labels, np.array(probs, copy=False)
#   替换为     return labels, np.asarray(probs)



if __name__ == '__main__':

    model_unsupervised = fasttext.train_unsupervised('../data/douban_comment_fixed.txt',epoch=10,dim=300)
    print(f'【盗墓笔记】相近的评论 {model_unsupervised.get_nearest_neighbors("盗墓笔记")}')
    print(f'【故事】相近 {model_unsupervised.get_nearest_neighbors("故事")}')
    print(f'【神话】相近 {model_unsupervised.get_nearest_neighbors("神话")}')

    model_supervised = fasttext.train_supervised('../data/douban_comment_fixed.txt', epoch=10, dim=300)

    print(f' pred [盗墓笔记怎么样] {model_supervised.predict("盗墓笔记怎么样")}')
    writer = SummaryWriter('../data/logs_word2vec')

    meta = model_unsupervised.words

    embedding_word = []
    for word in meta:
        embedding_word.append(model_unsupervised.get_word_vector(word))

    # 词云显示
    writer.add_embedding(torch.tensor(embedding_word),metadata=meta)


    writer.flush()
    writer.close()
