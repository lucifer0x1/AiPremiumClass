
import fasttext
import torch
from torch.utils.tensorboard import SummaryWriter


if __name__ == '__main__':


    model_supervised = fasttext.train_supervised('../data/cooking.stackexchange.txt', epoch=10, dim=300)

    print(f' pred [oil for cooking eggs] {model_supervised.predict("oil for cooking eggs")}')
    writer = SummaryWriter('../data/logs_fasttext')

    meta = model_supervised.words
    embedding_word = []
    for word in meta:
        embedding_word.append(model_supervised.get_word_vector(word))
    # 词云显示
    writer.add_embedding(torch.tensor(embedding_word),metadata=meta)


    writer.flush()
    writer.close()
