import re
import jieba
from main import config
# 自定义词表
jieba.load_userdict(config.user_dict)
print('jiea loaded user dict')

remove_words = ['|', '[', ']', '语音', '图片']
def clean_sentence(sentence):
    '''
    特殊符号去除
    :param sentence: 待处理的字符串

    :return: 过滤特殊字符后的字符串
    '''
    if isinstance(sentence, str):
        return re.sub(
            r'[\s+\-\/\[\]\{\}_$%^*(+\"\')]+|[+——()【】“”~@#￥%……&*（）]',
            ' ', sentence)
    else:
        return ' '


def filter_words(sentence):
    '''
    过滤停用词
    :param seg_list: 切好词的列表 [word1 ,word2 .......]
    :return: 过滤后的停用词
    '''
    words = sentence.split(' ')
    # 去掉多余空字符
    words = [word for word in words if word and word not in remove_words]
    # 去掉停用词 包括一下标点符号也会去掉
    # words = [word for word in words if word not in stop_words]
    return words


def seg_proc(sentence):
    tokens = sentence.split('|')
    result = []
    for t in tokens:
        result.append(cut_sentence(t))
    return ' | '.join(result)


def cut_sentence(line):
    # 切词，默认精确模式，全模式cut参数cut_all=True
    tokens = jieba.cut(line)
    return ' '.join(tokens)


def sentence_proc(sentence):
    '''
    预处理模块
    :param sentence:待处理字符串
    :return: 处理后的字符串
    '''
    # 清除无用词
    sentence = clean_sentence(sentence)
    # 分段切词
    sentence = seg_proc(sentence)
    # 过滤停用词
    words = filter_words(sentence)

    # 拼接成一个字符串,按空格分隔
    return ''.join(words)


def sentence_proc2(sentence):
    '''
    预处理模块
    :param sentence:待处理字符串
    :return: 处理后的字符串
    '''
    # 清除无用词
    sentence = clean_sentence(sentence)
    # 分段切词
    sentence = seg_proc(sentence)
    # 过滤停用词
    words = filter_words(sentence)

    # 拼接成一个字符串,按空格分隔
    return ' '.join(words)


def sentences_proc(df):
    '''
    数据集批量处理方法
    :param df: 数据集
    :return:处理好的数据集

    '''
    # 批量预处理 训练集和测试集
    for col_name in ['Brand', 'Model', 'Question', 'Dialogue']:
        print(col_name)
        df[col_name] = df[col_name].apply(sentence_proc)

    if 'Report' in df.columns:
        # 训练集 Report 预处理
        df['Report'] = df['Report'].apply(sentence_proc)
    return df


def sentences_proc2(df):
    '''
    数据集批量处理方法
    :param df: 数据集
    :return:处理好的数据集

    '''
    # 批量预处理 训练集和测试集
    for col_name in ['Brand', 'Model', 'Question', 'Dialogue']:
        print(col_name)
        df[col_name] = df[col_name].apply(sentence_proc2)

    if 'Report' in df.columns:
        # 训练集 Report 预处理
        df['Report'] = df['Report'].apply(sentence_proc2)
    return df

