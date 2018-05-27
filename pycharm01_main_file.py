import pandas as pd
import numpy as np
import time
from pycharm02_feature_engineering import read_data, data_dummies, feature_engineer, is_weekend
from pycharm03_gbdt_features import gbdt_model
import warnings
np.random.seed(1)
warnings.filterwarnings('ignore')


from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten

import lightgbm as lgb

if __name__ == '__main__':
    data, train, test = read_data('./data/round1_ijcai_18_train_20180301.txt', \
                                  './data/round1_ijcai_18_test_b_20180418.txt')

    data, train, test = feature_engineer(data, train, test)
    print('length of data columns: ', len(data.columns))

    # 根据每天记录的条目设定是否为周末
    data = is_weekend(data)

    # 特征选取
    base_features = ['instance_id']
    numerical_features = ['shop_review_positive_rate', 'shop_score_service', 'shop_score_delivery',
                          'shop_score_description', 'item_id_history_ctr', 'shop_id_history_ctr',
                          'user_query_day', 'item_query_day', 'shop_query_day',

                          ]
    # 'user_query_day_hour', 'item_query_day_hour', 'shop_query_day_hour',
    # 'user_item_display_day', 'item_shop_display_day', 'user_shop_display_day',
    # 'user_item_display_hour', 'shop_item_display_hour', 'shop_user_display_hour',
    # 'shop_query_day', 'shop_query_day_hour',
    # 'shop_item_display_day', 'shop_item_display_hour'
    # 其他特征
    other_features = ['day']
    #  'item_id', 'user_id', 'context_id', 'shop_id',  这些特征如果one-hot编码，内存吃不下
    categorical_features = ['item_category_list', 'item_city_id',
                            'item_price_level', 'item_sales_level',
                            'item_collected_level', 'item_pv_level', 'user_gender_id', 'user_age_level',
                            'user_occupation_id', 'user_star_level', 'hour', 'context_page_id',
                            'shop_review_num_level', 'shop_star_level', 'is_weekend']
    non_cate_features = numerical_features + base_features + other_features
    # 类别特征one-hot编码
    new_data = data_dummies(data, categorical_features, non_cate_features)


    #gbdt_features = gbdt_model(data, numerical_features)
    #new_data = pd.concat([new_data, gbdt_features], axis=1)
    print('final data shape: ', new_data.shape)

    # 目标变量
    new_data['is_trade'] = data['is_trade']

    print('DataSet shape(after feature engineer: ', new_data.shape)

    print('--------------------------------------------')

    print('info of DataSet shape(after feature engineer: \n', new_data.info())

    print('--------------------------------------------')

    # 划分数据集
    # train = new_data[new_data.day < 24]
    # test = new_data[new_data.day == 24]
    all_train = new_data[new_data.day < 25]
    predict = new_data[new_data.day == 25]


    features = []
    target = 'is_trade'
    for feat in all_train.columns:
        if feat not in ['instance_id', 'item_id', 'shop_id', 'day', 'is_trade', 'context_id']:
            features.append(feat)

    X_train, X_test, y_train, y_test = train_test_split(all_train[features], all_train[target], test_size=0.3, random_state=10)
    # X_train = train[features]
    # y_train = train[target]
    # X_test = test[features]
    # y_test = test[target]

    model_flag = ['lr']
    for mf in model_flag:
        if mf == 'lr':
            print('start Logistic Regression molde training...')
            clf = LogisticRegression(penalty='l2', C=1.0, max_iter=800, random_state=1)
            kf = KFold(n_splits=4, shuffle=True, random_state=10)
            for train_idx, test_idx in kf.split(all_train):
                train = all_train.iloc[train_idx, :]
                test = all_train.iloc[test_idx, :]
                X_train = train[features]
                y_train = train[target]
                X_test = test[features]
                y_test = test[target]

                clf.fit(X_train, y_train)
                X_test['predicted_score'] = clf.predict_proba(X_test, )[:, 1]
                X_test['predicted_score'] = X_test['predicted_score']
                X_test['predicted_score'] = X_test['predicted_score'].apply(lambda x: 0 if x < 0 else x)
                log_loss_value = log_loss(y_test, X_test['predicted_score'])
                print('The log loss value of lr model: ', log_loss_value)
            #
            clf.fit(all_train[features], all_train[target])
            predict['predicted_score'] = clf.predict_proba(predict[features])[:, 1]
            predict['predicted_score'] = predict['predicted_score']
            predict[['instance_id', 'predicted_score']].to_csv('result4_21_lr.csv', index=False, sep=' ') # , float_format='%.15f'

            # L1的预测值会比L2的普遍更大一些。L2训练速度更快
            # C=1.0 0.08745958476513374  预测集的logloss: 0.08766  减去0.003后测试集logloss：0.0879364704475769
            # 增加是否是周末('is_weekend')：0.08742989510612711(有一定的提升)
            # 增加（'user_query_day', 'user_query_day_hour'）: 0.08678697715978508(有提升)
            # 增加（'user_item_display_day', 'user_item_display_hour'）: 0.08659428243529521(有提升)
            # 增加（'shop_query_day', 'shop_query_day_hour'）: 0.08669528212926571（下降）
            # 增加（'shop_item_display_day', 'shop_item_display_hour'）: 0.08665409076544187
            # 0-1标准化以后：0.08659754069041278 --去掉shop的统计特征--> 0.08659569868563805(相比不标准化，性能下降)
            # 预测后减去0.001，logloss为0.086577893296759（有提升）  减去0.002：0.086860847028705
            #                         0.08656245651907789
            #                         0.08657047727935244
            #                         0.08629663083740087 0.08593615286434565 0.08439167847165951 0.08438629840705225


        if mf == 'dnn':
            rows, cols = all_train[features].shape
            model = Sequential()
            model.add(Dense(128, input_dim=cols, activation='relu'))
            model.add(Dropout(0.5))
            # model.add(Dense(128, activation='relu'))
            # model.add(Dropout(0.6))
            # model.add(Dense(64, activation='relu'))
            # model.add(Dropout(0.7))
            model.add(Dense(64, activation='relu'))
            model.add(Dropout(0.8))
            model.add(Dense(32, activation='relu'))
            model.add(Dropout(0.9))
            model.add(Dense(16, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(optimizer='RMSprop', loss='binary_crossentropy', metrics=['accuracy'])

            model.fit(np.array(X_train), np.array(y_train), nb_epoch=20, batch_size=128)

            # #score = model.evaluate(np.array(test[features]), np.array(test[target]), batch_size=128)
            X_test['predicted_score'] = model.predict_proba(np.array(X_test), batch_size=128)
            logloss = log_loss(y_test, X_test['predicted_score'])
            print(logloss)  # 0.11677781879887103

            model.fit(all_train[features], all_train[target])
            predict['predicted_score'] = model.predict_proba(predict[features])
            # predict[['instance_id', 'predicted_score']].to_csv('result4_18_dnn_2.csv', index=False, sep=' ')
            print(logloss)

        if mf == 'lgb':  ## gbdt算法在预测集上表现非常不好。
            log_loss_list = []
            kf = KFold(n_splits=5, shuffle=True, random_state=1)
            for train_idx, test_idx in kf.split(all_train):
                train = all_train.iloc[train_idx, :]
                test = all_train.iloc[test_idx, :]
                X_train = train[features]
                y_train = train[target]
                X_test = test[features]
                y_test = test[target]

                clf = lgb.LGBMClassifier(num_leaves=65, max_depth=6, n_estimators=150, n_jobs=20, learning_rate=0.06, lambda_l2=1.0)
                clf.fit(X_train, y_train, feature_name=features)
                X_test['predicted_score'] = clf.predict_proba(X_test, )[:, 1]
                # X_test['predicted_score'] = X_test['predicted_score'] - 0.001
                log_loss_value = log_loss(y_test, X_test['predicted_score'])
                log_loss_list.append(log_loss_value)
            print('the log loss of lgb model in cv with 5 splits: ', log_loss_list)

            # 0.08270811820531722
            # (num_leaves=50, max_depth=5, n_estimators=120, n_jobs=20): 0.08258190767889406
            # (num_leaves=50, max_depth=5, n_estimators=150, n_jobs=20, learning_rate=0.1, num_iterators=1000): 0.08258016897053531
            # (num_leaves=36, max_depth=5, n_estimators=150, n_jobs=20, learning_rate=0.05, lambda_l2=1.0):     0.08255235859456307
            # (num_leaves=65, max_depth=6, n_estimators=150, n_jobs=20, learning_rate=0.05, lambda_l2=1.0):     0.08252046284038406
            # (num_leaves=65, max_depth=6, n_estimators=150, n_jobs=20, learning_rate=0.06, lambda_l2=1.0):     0.08243291190872347
            # (num_leaves=65, max_depth=6, n_estimators=150, n_jobs=20, learning_rate=0.07, lambda_l2=1.0):     0.08246869876021136
            # (num_leaves=65, max_depth=6, n_estimators=150, n_jobs=20, learning_rate=0.08, lambda_l2=1.0):     0.08250570813361169
            # (num_leaves=65, max_depth=6, n_estimators=150, n_jobs=20, learning_rate=0.1, lambda_l2=1.0):      0.08261049566089193
            # (num_leaves=129, max_depth=7, n_estimators=150, n_jobs=20, learning_rate=0.05, lambda_l2=1.0):    0.08258565875008496

            # CV: (num_leaves=65, max_depth=6, n_estimators=150, n_jobs=20, learning_rate=0.06, lambda_l2=1.0)
            # ==> [0.08182736225579536, 0.08345970586925804, 0.0832755139996111, 0.08366608032774034, 0.08265013401854969]
            # ==> 该模型预测集的logloss为 0.09860
            # 结果减去0.001
            # ==> [0.08183903852524796, 0.08346966702100868, 0.08328359755003954, 0.08365504210316921, 0.08265924934632558]


            clf = lgb.LGBMClassifier(num_leaves=65, max_depth=6, n_estimators=150, n_jobs=20, learning_rate=0.06,
                                     lambda_l2=1.0)
            clf.fit(all_train[features], all_train[target], feature_name=features)
            predict['predicted_score'] = clf.predict_proba(predict[features])[:, 1]
            # predict['predicted_score'] = predict['predicted_score'] - 0.002
            # predict[['instance_id', 'predicted_score']].to_csv('result4_19_lgb.csv', index=False, sep=' ', float_format='%.19f') # , float_format='%.15f'
