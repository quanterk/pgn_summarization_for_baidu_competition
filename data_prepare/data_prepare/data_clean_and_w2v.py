# 1.加载数据
import numpy as np
import pandas as pd
import os
base_dir = os.path.abspath(os.path.dirname(__file__))
#print(base_dir)
import sys
root_dir = base_dir+"/../"
sys.path.append(root_dir)
from main import config
from tools.multi_proc_utils import parallelize
from tools.raw_data_clean import sentences_proc, sentences_proc2



from multiprocessing import cpu_count
cores = cpu_count()

data_path = config.root_dir
print(data_path)
#未清洗数据
train_path=data_path+'AutoMaster_TrainSet.csv'
test_path=data_path+'AutoMaster_TestSet.csv'
ARTICLE_FILE = data_path+"train_text.txt"
SUMMARRY_FILE = data_path+"train_label.txt"
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
print('train data size {},test data size {}'.format(len(train_df), len(test_df)))

# 3.多线程, 批量数据处理，就是清洗文本，不分词
train_df = parallelize(train_df, sentences_proc)
test_df = parallelize(test_df, sentences_proc)
train_path2=data_path+'AutoMaster_TrainSet2_clean.csv'
test_path2=data_path+'AutoMaster_TestSet2_clean.csv'
train_df.to_csv(train_path2, index=None)
test_df.to_csv(test_path2, index=None)



# 3.多线程, 批量数据处理,用来做词向量，按空格分开的
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
print('train data size {},test data size {}'.format(len(train_df), len(test_df)))
train_df = parallelize(train_df, sentences_proc2)
test_df = parallelize(test_df, sentences_proc2)

# 4. 合并训练测试集合
train_df['merged'] = train_df[['Question', 'Dialogue', 'Report']].apply(lambda x: ' '.join(x), axis=1)
test_df['merged'] = test_df[['Question', 'Dialogue']].apply(lambda x: ' '.join(x), axis=1)
merged_df = pd.concat([train_df[['merged']], test_df[['merged']]], axis=0)
print('train data size {},test data size {},merged_df data size {}'.format(len(train_df),len(test_df), len(merged_df)))
train_df.to_csv(data_path+'split_train.csv')
merger_seg_path = data_path+'merged_train_test.csv'

# 6. 保存合并数据
merged_df.to_csv(merger_seg_path, index=None, header=False)


merger_seg_path = data_path+'merged_train_test.csv'

from gensim.models.word2vec import LineSentence, Word2Vec
#from gensim.models.word2vec import LineSentence, Word2Vec

# 7. 训练词向量
print('start build w2v model')
wv_model = Word2Vec(LineSentence(merger_seg_path),
                    size=300,
                    sg=1,
                    workers=cores,
                    iter=config.w2v_iter,
                    window=5,
                    min_count=5)


# 12. 更新vocab
vocab = {word: index for index, word in enumerate(wv_model.wv.index2word)}
reverse_vocab = {index: word for index, word in enumerate(wv_model.wv.index2word)}

print("Writing vocab file...")
with open(os.path.join(data_path, "vocab"), 'w', encoding='utf-8') as writer:
    for word, count in vocab.items():
        writer.write(word + ' ' + str(count) + '\n')


## 增添几个特殊字符的[unk],[pad] embedding,保存系数矩阵
embedding_matrix = wv_model.wv.vectors
pad = embedding_matrix[0:4]/4
r = np.vstack((pad,embedding_matrix))
np.save(config.embedding_matrix_path, r)
print('word vector saved')