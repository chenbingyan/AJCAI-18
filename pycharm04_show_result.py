import pandas as pd

result_lgb = pd.read_csv('./result4_19_lgb.csv', sep=' ', encoding='utf-8')
result_lgb = result_lgb.sort_values(by=['predicted_score'], ascending=False)
print(len(result_lgb))

result_lr = pd.read_csv('./result4_19_lr.csv', sep=' ', encoding='utf-8')
result_lr = result_lr.sort_values(by=['predicted_score'], ascending=False)
print(len(result_lr))

result = pd.merge(result_lgb, result_lr, on=['instance_id'], how='left')
result = result.reset_index(drop=True)
result = result.sort_values(by=['predicted_score_x'], ascending=False)
print(result.head(100))