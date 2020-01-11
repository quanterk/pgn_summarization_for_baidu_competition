def get_max_len(data):
    """
    获得合适的最大长度值
    :param data: 待统计的数据  train_df['Question']
    :return: 最大长度值
    """
    # TODO FIX len size bug
    max_lens = data.apply(lambda x: x.count(' ') + 1)
    return int(np.mean(max_lens) + 2 * np.std(max_lens))
