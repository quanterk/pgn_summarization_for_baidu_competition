import numpy as np
import pandas as pd
import os
import pathlib
base_dir = os.path.abspath(os.path.dirname(__file__))
#print(base_dir)
import sys
root_dir = base_dir+"/../"
sys.path.append(root_dir)

from main import  config


def write_values(values, path):
    with open(path, 'w') as f:
        for ar in values:
            f.write(ar + '\n')

# 获取项目根目录
# root = pathlib.Path(os.path.abspath(__file__)).parent.parent

data_path = config.root_dir
train_path = data_path + 'AutoMaster_TrainSet2_clean.csv'

ARTICLE_FILE = data_path + "train_text.txt"
SUMMARRY_FILE = data_path + "train_label.txt"
#
# # 训练数据路径
# train_path = os.path.join(root, 'data', 'AutoMaster_TrainSet.csv')
# # 测试数据路径
# test_data_path = os.path.join(root, 'data', 'AutoMaster_TestSet.csv')
# ARTICLE_FILE = os.path.join(root, 'data', "train_text.txt")
# SUMMARRY_FILE = os.path.join(root, 'data', "train_label.txt")
train = pd.read_csv(train_path)

train = train.dropna()

train['merged'] = train[['Question', 'Dialogue']].apply(lambda x: ' '.join(x), axis=1)

write_values(train['merged'].values, ARTICLE_FILE)
write_values(train['Report'].values, SUMMARRY_FILE)
