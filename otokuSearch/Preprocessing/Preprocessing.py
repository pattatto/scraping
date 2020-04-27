import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn import preprocessing
import pandas_profiling as pdp

df = pd.read_csv('otokuSearch/data/suumo.csv', sep='\t', encoding='utf-16')

splitted1 = df['立地1'].str.split(' 歩', expand=True)
splitted1.columns = ['立地11', '立地12']
splitted2 = df['立地2'].str.split(' 歩', expand=True)
splitted2.columns = ['立地21', '立地22']
splitted3 = df['立地3'].str.split(' 歩', expand=True)
splitted3.columns = ['立地31', '立地32']

splitted4 = df['敷/礼/保証/敷引,償却'].str.split('/', expand=True)
splitted4.columns = ['敷金', '礼金']

df = pd.concat([df, splitted1, splitted2, splitted3, splitted4], axis=1)

df.drop(['立地1','立地2','立地3','敷/礼/保証/敷引,償却'], axis=1, inplace=True)

df = df.dropna(subset=['賃料料'])

df['賃料料'] = df['賃料料'].str.replace(u'万円', u'')
df['敷金'] = df['敷金'].str.replace(u'万円', u'')
df['礼金'] = df['礼金'].str.replace(u'万円', u'')
df['管理費'] = df['管理費'].str.replace(u'円', u'')
df['築年数'] = df['築年数'].str.replace(u'新築', u'0')
df['築年数'] = df['築年数'].str.replace(u'99年以上', u'0') #
df['築年数'] = df['築年数'].str.replace(u'築', u'')
df['築年数'] = df['築年数'].str.replace(u'年', u'')
df['専有面積'] = df['専有面積'].str.replace(u'm', u'')
df['立地12'] = df['立地12'].str.replace(u'分', u'')
df['立地22'] = df['立地22'].str.replace(u'分', u'')
df['立地32'] = df['立地32'].str.replace(u'分', u'')

df['管理費'] = df['管理費'].replace('-',0)
df['敷金'] = df['敷金'].replace('-',0)
df['礼金'] = df['礼金'].replace('-',0)

splitted5 = df['立地11'].str.split('/', expand=True)
splitted5.columns = ['路線1', '駅1']
splitted5['駅徒歩1'] = df['立地12']
splitted6 = df['立地21'].str.split('/', expand=True)
splitted6.columns = ['路線2', '駅2']
splitted6['駅徒歩2'] = df['立地22']
splitted7 = df['立地31'].str.split('/', expand=True)
splitted7.columns = ['路線3', '駅3']
splitted7['駅徒歩3'] = df['立地32']

df = pd.concat([df, splitted5, splitted6, splitted7], axis=1)

df.drop(['立地11','立地12','立地21','立地22','立地31','立地32'], axis=1, inplace=True)

df['賃料料'] = pd.to_numeric(df['賃料料'])
df['管理費'] = pd.to_numeric(df['管理費'])
df['敷金'] = pd.to_numeric(df['敷金'])
df['礼金'] = pd.to_numeric(df['礼金'])
df['築年数'] = pd.to_numeric(df['築年数'])
df['専有面積'] = pd.to_numeric(df['専有面積'])

df['賃料料'] = df['賃料料'] * 10000
df['敷金'] = df['敷金'] * 10000
df['礼金'] = df['礼金'] * 10000

df['駅徒歩1'] = pd.to_numeric(df['駅徒歩1'])
df['駅徒歩2'] = pd.to_numeric(df['駅徒歩2'])
df['駅徒歩3'] = pd.to_numeric(df['駅徒歩3'])

splitted8 = df['階層'].str.split('-', expand=True)
splitted8.columns = ['階1', '階2']
splitted8['階1'].str.encode('cp932')
splitted8['階1'] = splitted8['階1'].str.replace(u'階', u'')
splitted8['階1'] = splitted8['階1'].str.replace(u'B', u'-')
splitted8['階1'] = splitted8['階1'].str.replace(u'M', u'')
splitted8['階1'] = pd.to_numeric(splitted8['階1'])
df = pd.concat([df, splitted8], axis=1)

df['建物の高さ'] = df['建物の高さ'].str.replace(u'地下1地上', u'')
df['建物の高さ'] = df['建物の高さ'].str.replace(u'地下2地上', u'')
df['建物の高さ'] = df['建物の高さ'].str.replace(u'地下3地上', u'')
df['建物の高さ'] = df['建物の高さ'].str.replace(u'地下4地上', u'')
df['建物の高さ'] = df['建物の高さ'].str.replace(u'地下5地上', u'')
df['建物の高さ'] = df['建物の高さ'].str.replace(u'地下6地上', u'')
df['建物の高さ'] = df['建物の高さ'].str.replace(u'地下7地上', u'')
df['建物の高さ'] = df['建物の高さ'].str.replace(u'地下8地上', u'')
df['建物の高さ'] = df['建物の高さ'].str.replace(u'地下9地上', u'')
df['建物の高さ'] = df['建物の高さ'].str.replace(u'平屋', u'1')
df['建物の高さ'] = df['建物の高さ'].str.replace(u'階建', u'')
df['建物の高さ'] = pd.to_numeric(df['建物の高さ'])

df = df.reset_index(drop=True)
df['間取りDK'] = 0
df['間取りK'] = 0
df['間取りL'] = 0
df['間取りS'] = 0
df['間取り'] = df['間取り'].str.replace(u'ワンルーム', u'1')

for x in range(len(df)):
    if 'DK' in df['間取り'][x]:
        df.loc[x,'間取りDK'] = 1
df['間取り'] = df['間取り'].str.replace(u'DK',u'')

for x in range(len(df)):
    if 'K' in df['間取り'][x]:
        df.loc[x,'間取りK'] = 1
df['間取り'] = df['間取り'].str.replace(u'K',u'')

for x in range(len(df)):
    if 'L' in df['間取り'][x]:
        df.loc[x,'間取りL'] = 1
df['間取り'] = df['間取り'].str.replace(u'L',u'')

for x in range(len(df)):
    if 'S' in df['間取り'][x]:
        df.loc[x,'間取りS'] = 1
df['間取り'] = df['間取り'].str.replace(u'S',u'')

df['間取り'] = pd.to_numeric(df['間取り'])

splitted9 = df['住所'].str.split('区', expand=True)
splitted9.columns = ['市町村']
#splitted9['区'] = splitted9['区'] + '区'
#splitted9['区'] = splitted9['区'].str.replace('東京都','')
df = pd.concat([df, splitted9], axis=1)

splitted10 = df['駅1'].str.split(' バス', expand=True)
splitted10.columns = ['駅1', 'バス1']
splitted11 = df['駅2'].str.split(' バス', expand=True)
splitted11.columns = ['駅2', 'バス2']
splitted12 = df['駅3'].str.split(' バス', expand=True)
splitted12.columns = ['駅3', 'バス3']

splitted13 = splitted10['バス1'].str.split('分 \(バス停\)', expand=True)
splitted13.columns = ['バス時間1', 'バス停1']
splitted14 = splitted11['バス2'].str.split('分 \(バス停\)', expand=True)
splitted14.columns = ['バス時間2', 'バス停2']
splitted15 = splitted12['バス3'].str.split('分 \(バス停\)', expand=True)
splitted15.columns = ['バス時間3', 'バス停3']

splitted16 = pd.concat([splitted10, splitted11, splitted12, splitted13, splitted14, splitted15], axis=1)
splitted16.drop(['バス1','バス2','バス3'], axis=1, inplace=True)

df.drop(['駅1','駅2','駅3'], axis=1, inplace=True)
df = pd.concat([df, splitted16], axis=1)

splitted17 = df['駅1'].str.split(' 車', expand=True)
splitted17.columns = ['駅1', '車1']
splitted18 = df['駅2'].str.split(' 車', expand=True)
splitted18.columns = ['駅2', '車2']
splitted19 = df['駅3'].str.split(' 車', expand=True)
splitted19.columns = ['駅3', '車3']

splitted20 = splitted17['車1'].str.split('分', expand=True)
splitted20.columns = ['車時間1', '車距離1']
splitted21 = splitted18['車2'].str.split('分', expand=True)
splitted21.columns = ['車時間2', '車距離2']
splitted22 = splitted19['車3'].str.split('分', expand=True)
splitted22.columns = ['車時間3', '車距離3']

splitted23 = pd.concat([splitted17, splitted18, splitted19, splitted20, splitted21, splitted22], axis=1)
splitted23.drop(['車1','車2','車3'], axis=1, inplace=True)

df.drop(['駅1','駅2','駅3'], axis=1, inplace=True)
df = pd.concat([df, splitted23], axis=1)

df['車距離1'] = df['車距離1'].str.replace(u'\(', u'')
df['車距離1'] = df['車距離1'].str.replace(u'km\)', u'')
df['車距離2'] = df['車距離2'].str.replace(u'\(', u'')
df['車距離2'] = df['車距離2'].str.replace(u'km\)', u'')
df['車距離3'] = df['車距離3'].str.replace(u'\(', u'')
df['車距離3'] = df['車距離3'].str.replace(u'km\)', u'')

df[['路線1','路線2','路線3', '駅1', '駅2','駅3','市町村', 'バス停1', 'バス停2', 'バス停3']] = df[['路線1','路線2','路線3', '駅1', '駅2','駅3','市町村', 'バス停1', 'バス停2', 'バス停3']].fillna("NAN")
df[['バス時間1','バス時間2','バス時間3',]] = df[['バス時間1','バス時間2','バス時間3']].fillna(0)#欠損値あると特徴量の計算でエラーがでるため０に置換
df['バス時間1'] = df['バス時間1'].astype(float)
df['バス時間2'] = df['バス時間2'].astype(float)
df['バス時間3'] = df['バス時間3'].astype(float)

oe = preprocessing.OrdinalEncoder()
df[['建物種別', '路線1','路線2','路線3', '駅1', '駅2','駅3','市町村', 'バス停1', 'バス停2', 'バス停3']] = oe.fit_transform(df[['建物種別', '路線1','路線2','路線3', '駅1', '駅2','駅3','市町村', 'バス停1', 'バス停2', 'バス停3']].values)

df['賃料料+管理費'] = df['賃料料'] + df['管理費']

df_for_search = df.copy()

#上限価格を設定
df = df[df['賃料料+管理費'] < 300000]

df = df[["マンション名",'建物種別', '賃料料+管理費', '築年数', '建物の高さ', '階1',
       '専有面積','路線1','路線2','路線3', '駅1', '駅2','駅3','駅徒歩1', '駅徒歩2','駅徒歩3','間取り', '間取りDK', '間取りK', '間取りL', '間取りS',
       '市町村', 'バス停1', 'バス停2', 'バス停3', 'バス時間1','バス時間2','バス時間3']]

df.columns = ['name', 'building', 'real_rent','age', 'hight', 'level','area', 'route_1','route_2','route_3','station_1','station_2','station_3','station_wolk_1','station_wolk_2','station_wolk_3','room_number','DK','K','L','S','adress', 'bus_stop1', 'bus_stop2', 'bus_stop3', 'bus_time1', 'bus_time2', 'bus_time3']


#pdp.ProfileReport(df)
df.to_csv('otokuSearch/Preprocessing/Preprocessing.csv', sep = '\t', encoding='utf-16', header=True, index=False)
df_for_search.to_csv('otokuSearch/Preprocessing/df_for_search.csv', sep = '\t', encoding='utf-16', header=True, index=False)
