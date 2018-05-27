import pandas as pd
import numpy as np
import time
from collections import Counter
import datetime
#from pycharm03_gbdt_features import *


# read data
def read_data(train_data_path, test_data_path):
    train = pd.read_csv(train_data_path, sep=' ', encoding='utf-8')
    test = pd.read_csv(test_data_path, sep=' ', encoding='utf-8')
    train.drop_duplicates(inplace=True)
    test['is_trade'] = np.nan
    print('shape of train data: ', train.shape)
    print('shape of test data: ', test.shape)
    # 将训练集和测试集拼接在一起，方便特征处理
    data = pd.concat([train, test], axis=0)
    print('shape of whole data: ', data.shape)
    return data, train, test

# 处理时间数据
def timestamp_datetime(value):
    format = '%Y-%m-%d %H:%M:%S'
    dt = time.localtime(value)
    dt = time.strftime(format, dt)
    dt = datetime.datetime.strptime(dt, format)
    return dt

# 时间映射
def time_map(data):
    data['time'] = data.context_timestamp.apply(timestamp_datetime)
    data['day'] = data.time.apply(lambda x: int(x.day))
    data['hour'] = data.time.apply(lambda x: int(x.hour))
    return data

## 构建新特征1: item的历史CTR
# 计算历史CTR(计算从18号-24号每件商品：发生交易的总次数/该商品在数据集中的曝光次数)
## 构建新特征2：shop的历史CTR
def history_ctr(train_data, feature):
    '''
    # item_trade_data = train_data[['item_id', 'is_trade']]
    item_id_df = pd.DataFrame(train_data.item_id.unique()).reset_index(drop=True).rename(columns={0: 'item_id'})
    item_id_list = train_data.item_id.unique()
    history_ctr_list = []
    for item in item_id_list:
        display_data = train_data[train_data.item_id == item]
        # 为了避免某些商品展示次数很低，但却有点击，导致历史CTR过高而不真实，故而需要设定阈值干预处理
        history_ctr_value = np.sum(display_data['is_trade']) / len(display_data)
        if len(display_data) < 100 and history_ctr_value > 0.01:
            history_ctr_value = 0.01
        history_ctr_list.append(history_ctr_value)
    item_id_df['history_ctr'] = history_ctr_list
    return item_id_df
    '''
    feature_df = pd.DataFrame(train_data[feature].unique()).reset_index(drop=True).rename(columns={0: feature})
    feature_list = train_data[feature].unique()
    history_ctr_list = []
    for val in feature_list:
        display_data = train_data[train_data[feature] == val]
        history_ctr_value = np.sum(display_data['is_trade']) / len(display_data)
        if len(display_data) < 50 and history_ctr_value > 0.1:
            # 0.09 -> 0.08366686072604275   0.08 -> 0.08379288681304232   0.05 -> 0.08438629840705225
            history_ctr_value = 0.1
        history_ctr_list.append(history_ctr_value)
    new_feature = feature + '_history_ctr'
    feature_df[new_feature] = history_ctr_list
    print(feature , ': ', history_ctr_list[0:10])
    return feature_df

# 根据数据集中数据条目判断是否时周末，预测集的处理应该把两个阶段的数据进行合并统计
def is_weekend(data):
    instance_count = data['day'].value_counts()
    instance_count = pd.DataFrame(instance_count)
    instance_count['is_weekend'] = instance_count['day'].apply(lambda x: 1 if x > 68000 else 0)
    instance_count = instance_count.reset_index().rename(columns={'index':'day', 'day': 'day_count'})
    del instance_count['day_count']
    data = pd.merge(data, instance_count, on='day', how='left')
    return data

# 对类别型变量进行one-hot encode
def data_dummies(data, cate_feat_list, base_features):
    print('start one-hot encode......')
    new_data = data[base_features]
    for feat in cate_feat_list:
        categorical_data = data[feat].astype('str')
        categorical_data = pd.get_dummies(categorical_data, dummy_na=True, prefix=feat)
        new_data = pd.concat([new_data, categorical_data], axis=1)
        print(feat, ' has finished!')
    print('all categorical features have been one-hot encoded!')
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    return new_data

# 特征工程函数
def feature_engineer(data, train, test):
    # 将timestamp映射出年月日，然后提取日期、小时
    data = time_map(data)

    new_feat = []
    ###
    user_query_day = data.groupby(['user_id', 'day']).size().reset_index().rename(columns={0: 'user_query_day'})
    data = pd.merge(data, user_query_day, 'left', on=['user_id', 'day'])
    new_feat.append('user_query_day')
    ###
    item_query_day = data.groupby(['item_id', 'day']).size().reset_index().rename(columns={0: 'item_query_day'})
    data = pd.merge(data, item_query_day, 'left', on=['item_id', 'day'])
    new_feat.append('item_query_day')
    ###
    shop_query_day = data.groupby(['shop_id', 'day']).size().reset_index().rename(columns={0: 'shop_query_day'})
    data = pd.merge(data, shop_query_day, on=['shop_id', 'day'], how='left')
    new_feat.append('shop_query_day')
    ###
    user_query_day_hour = data.groupby(['user_id', 'day', 'hour']).size().reset_index().rename(columns={0: 'user_query_day_hour'})
    data = pd.merge(data, user_query_day_hour, 'left', on=['user_id', 'day', 'hour'])
    new_feat.append('user_query_day_hour')
    ###
    item_query_day_hour = data.groupby(['item_id', 'day', 'hour']).size().reset_index().rename(columns={0: 'item_query_day_hour'})
    data = pd.merge(data, item_query_day_hour, 'left', on=['item_id', 'day', 'hour'])
    new_feat.append('item_query_day_hour')
    ###
    shop_query_day_hour = data.groupby(['shop_id', 'day', 'hour']).size().reset_index().rename(columns={0: 'shop_query_day_hour'})
    data = pd.merge(data, shop_query_day_hour, 'left', on=['shop_id', 'day', 'hour'])
    new_feat.append('shop_query_day_hour')
    ###
    user_item_query_day = data.groupby(['item_id', 'user_id', 'day']).size().reset_index().rename(columns={0: 'user_item_display_day'})
    data = pd.merge(data, user_item_query_day, on=['item_id', 'user_id', 'day'], how='left')
    new_feat.append('user_item_display_day')
    ###
    shop_item_query_day = data.groupby(['item_id', 'shop_id', 'day']).size().reset_index().rename(columns={0: 'item_shop_display_day'})
    data = pd.merge(data, shop_item_query_day, on=['item_id', 'shop_id', 'day'], how='left')
    new_feat.append('user_shop_display_day')
    ###
    user_shop_query_day = data.groupby(['shop_id', 'user_id', 'day']).size().reset_index().rename(columns={0: 'user_shop_display_day'})
    data = pd.merge(data, user_shop_query_day, on=['shop_id', 'user_id', 'day'], how='left')
    new_feat.append('user_shop_display_day')

    ###
    user_item_query_hour = data.groupby(['item_id', 'user_id', 'day', 'hour']).size().reset_index().rename(columns={0: 'user_item_display_hour'})
    data = pd.merge(data, user_item_query_hour, on=['item_id', 'user_id', 'day', 'hour'], how='left')
    new_feat.append('user_item_display_hour')
    ###
    shop_item_query_hour = data.groupby(['shop_id', 'item_id', 'day', 'hour']).size().reset_index().rename(columns={0: 'shop_item_display_hour'})
    data = pd.merge(data, shop_item_query_hour, on=['shop_id', 'item_id', 'day', 'hour'], how='left')
    new_feat.append('shop_item_display_hour')
    ###
    shop_user_query_hour = data.groupby(['shop_id', 'user_id', 'day', 'hour']).size().reset_index().rename(columns={0: 'shop_user_display_hour'})
    data = pd.merge(data, shop_user_query_hour, on=['shop_id', 'user_id', 'day', 'hour'], how='left')
    new_feat.append('shop_user_display_hour')


    for f in new_feat:
        data[f] = (data[f] - data[f].min()) / (data[f].max() - data[f].min())
    print(data.columns)
    print('************************')


    # 计算历史ctr
    for feat in ['item_id', 'shop_id']:
        history_ctr_df = history_ctr(train, feat)
        # 将历史ctr合并到数据集中
        data = pd.merge(data, history_ctr_df, on=feat, how='left')
    # 对history_ctr的缺失值补0.005(防止冷启动问题)--> 其实并没有缺失值，因为history_ctr_df包含了feature的所有取值。但是对于预测集需要防备
        #new_feature = feat + '_history_ctr'
        data[history_ctr_df.columns[1]].fillna(0.05, inplace=True)
    # 对于item_category_list，由于第一个类别都一样，第三个类别只有少量几个样本有，所以只取第二个类别
    data['item_category_list'] = data['item_category_list'].apply(lambda x: x.strip().split(';')[1])

    return data, train, test