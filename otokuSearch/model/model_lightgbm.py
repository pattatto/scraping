#データ解析用ライブラリ
import pandas as pd
import numpy as np

#データ可視化ライブラリ
import matplotlib.pyplot as plt
import seaborn as sns

#ランダムフォレストライブラリ
import lightgbm as lgb

#交差検証用に訓練データとモデル評価用データに分けるライブラリ
from sklearn.model_selection import KFold

#関数の処理で必要なライブラリ
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
#モデルの保存に必要なライブラリ
import pickle

#予測値と正解値を描写する関数
def True_Pred_map(pred_df):
    RMSE = np.sqrt(mean_squared_error(pred_df['true'], pred_df['pred']))
    R2 = r2_score(pred_df['true'], pred_df['pred'])
    plt.figure(figsize=(8,8))
    ax = plt.subplot(111)
    ax.scatter('true', 'pred', data=pred_df)
    ax.set_xlabel('True Value', fontsize=15)
    ax.set_ylabel('Pred Value', fontsize=15)
    ax.set_xlim(pred_df.min().min()-0.1 , pred_df.max().max()+0.1)
    ax.set_ylim(pred_df.min().min()-0.1 , pred_df.max().max()+0.1)
    x = np.linspace(pred_df.min().min()-0.1, pred_df.max().max()+0.1, 2)
    y = x
    ax.plot(x,y,'r-')
    plt.text(0.1, 0.9, 'RMSE = {}'.format(str(round(RMSE, 5))), transform=ax.transAxes, fontsize=15)
    plt.text(0.1, 0.8, 'R^2 = {}'.format(str(round(R2, 5))), transform=ax.transAxes, fontsize=15)


df = pd.read_csv('otokuSearch/Featurevalue/Fettur_evalue.csv', sep='\t', encoding='utf-16')

#kf : データ分割の挙動を指定する箱。今回は10分割・データシャッフルあり。
kf = KFold(n_splits=10, shuffle=True, random_state=1)

#predicted_df : これから各予測値を結合していく時に、空のデータフレームを容易しておく
predicted_df = pd.DataFrame({'index':0, 'pred':0}, index=[1])

#パラメータは特に調整してない
lgbm_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves':80
}

#交差検証4分割で行うので、10回ループが繰り返される。
#kfにindexを与え、訓練データindexと評価データindexを決定してもらう。
#df,indexの中から1回目につかう訓練用のデータのインデックス番号と評価用データのインデックス番号をtrain_index, val_indexに出力する
for train_index, val_index in kf.split(df.index):

    #訓練データindexと評価データindexを使って、訓練データと評価データ＆説明変数と目的変数に分割
    X_train = df.drop(['real_rent','name'], axis=1).iloc[train_index]
    y_train = df['real_rent'].iloc[train_index]
    X_test = df.drop(['real_rent','name'], axis=1).iloc[val_index]
    y_test = df['real_rent'].iloc[val_index]

    #LightGBM高速化の為のデータセットに加工する
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test)

    #LightGBMのモデル構築
    gbm = lgb.train(lgbm_params,
                lgb_train,
                valid_sets=(lgb_train, lgb_eval),
                num_boost_round=10000,
                early_stopping_rounds=100,
                verbose_eval=50)

    #モデルに評価用説明変数を入れて、予測値を出力する
    predicted = gbm.predict(X_test)

    #temp_df : 予測値と正答値を照合するために、予測値と元のindexを結合
    temp_df = pd.DataFrame({'index':X_test.index, 'pred':predicted})

    #predicted_df : 空のデータフレームにtemp_dfを結合→二周目以降のループでは、predicted_df（中身アリ）にtemp_dfが結合する
    predicted_df = pd.concat([predicted_df, temp_df], axis=0)

predicted_df = predicted_df.sort_values('index').reset_index(drop=True).drop(index=[0]).set_index('index')
predicted_df = pd.concat([predicted_df, df['real_rent']], axis=1).rename(columns={'real_rent' : 'true'})

True_Pred_map(predicted_df)

print(r2_score(y_test, predicted)  )
lgb.plot_importance(gbm, figsize=(12, 6))
plt.show()

#モデルの保存
with open('otokuSearch/model/model.pickle', mode='wb') as fp:
    pickle.dump(gbm, fp)
