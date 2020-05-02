import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle

#データの読み込み
df = pd.read_csv('otokuSearch/Featurevalue/Fettur_evalue.csv', sep='\t', encoding='utf-16')

#学習済モデルの読み込み
with open('otokuSearch/model/model.pickle', mode='rb') as fp:
    gbm = pickle.load(fp)


#お得物件データの作成
y = df["real_rent"]
X = df.drop(['real_rent',"name"], axis=1)
pred = list(gbm.predict(X, num_iteration=gbm.best_iteration))
pred = pd.Series(pred, name="予測値")
diff = pd.Series(df["real_rent"]-pred,name="予測値との差")
df_for_search = pd.read_csv('otokuSearch/Preprocessing/df_for_search.csv', sep='\t', encoding='utf-16')
df_for_search['賃料料+管理費'] = df_for_search['賃料料'] + df_for_search['管理費']
df_search = pd.concat([df_for_search,diff,pred], axis=1)
df_search = df_search.sort_values("予測値との差")
df_search = df_search[["マンション名",'賃料料+管理費', '予測値',  '予測値との差', '詳細URL', '間取り', '専有面積', '階層', '駅1', '駅徒歩1', '間取りDK', '間取りK', '間取りL']]
df_search.to_csv('otokuSearch/Otoku_data/otoku.csv', sep = '\t',encoding='utf-16')
