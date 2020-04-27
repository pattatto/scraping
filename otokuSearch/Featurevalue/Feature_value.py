import pandas as pd
import numpy as np

df = pd.read_csv('otokuSearch/Preprocessing/Preprocessing.csv', sep='\t', encoding='utf-16')

df["per_area"] = df["area"]/df["room_number"]
df["hight_level"] = df["hight"]*df["level"]
df["area_hight_level"] = df["area"]*df["hight_level"]
df["distance_staion_1"] = df["station_1"]*df["station_wolk_1"]+df["bus_stop1"]*df["bus_time1"]
df["distance_staion_2"] = df["station_2"]*df["station_wolk_2"]+df["bus_stop2"]*df["bus_time2"]
df["distance_staion_3"] = df["station_3"]*df["station_wolk_3"]+df["bus_stop3"]*df["bus_time3"]

df.to_csv('otokuSearch/Featurevalue/Fettur_evalue.csv', sep = '\t', encoding='utf-16', header=True, index=False)
