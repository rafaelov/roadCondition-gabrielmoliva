import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Abre os csvs es os transforma em dataframes
df_01 = pd.read_csv(r'pdfs/dataset_gps_01.csv', sep=',')
df_02 = pd.read_csv(r'pdfs/dataset_gps_02.csv', sep=',')
df_03 = pd.read_csv(r'pdfs/dataset_gps_03.csv', sep=',')
df_04 = pd.read_csv(r'pdfs/dataset_gps_04.csv', sep=',')
df_05 = pd.read_csv(r'pdfs/dataset_gps_05.csv', sep=',')
df_06 = pd.read_csv(r'pdfs/dataset_gps_06.csv', sep=',')
df_07 = pd.read_csv(r'pdfs/dataset_gps_07.csv', sep=',')
df_08 = pd.read_csv(r'pdfs/dataset_gps_08.csv', sep=',')
df_09 = pd.read_csv(r'pdfs/dataset_gps_09.csv', sep=',')

# Concatena todos os dataframes em um só
df_concat = pd.concat([df_01, df_02, df_03, df_04, df_05, 
                         df_06, df_07, df_08, df_09], axis=0)

# Remove colunas indesejadas
df_concat.drop(['latitude', 'longitude', 'elevation', 'accuracy', 'bearing', 'satellites', 'provider', 
                  'hdop', 'vdop', 'pdop', 'geoidheight', 'ageofdgpsdata', 'dgpsid', 'activity', 'battery',
                  'annotation'], axis=1, inplace=True)

# Remove colunas com valores não numéricos (NaN)
df_concat.dropna(inplace=True)

# Corrige a numeração do índice do dataframe
df_concat.reset_index(drop=True, inplace=True)

print(df_concat.info())