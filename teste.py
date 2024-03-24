import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import QuantileTransformer, OneHotEncoder
from sklearn.neighbors import KNeighborsRegressor
import seaborn as sb
import pandas.plotting as pp

#Abre o csv e transforma em um dataframe.
corsa_01 = r'opel_corsa_01.csv'
df_corsa_01 = pd.read_csv(corsa_01, sep=';')

corsa_02 = r'opel_corsa_02.csv'
df_corsa_02 = pd.read_csv(corsa_02, sep=';')

peugeot_01 = r'peugeot_207_01.csv'
df_peugeot_01 = pd.read_csv(peugeot_01, sep=';')

peugeot_02 = r'peugeot_207_02.csv'
df_peugeot_02 = pd.read_csv(peugeot_02, sep=';')

#Concatena todos os dataframes em um único.
df = pd.concat([df_corsa_01, df_corsa_02, df_peugeot_01, df_peugeot_02], axis=0)

#Renomea colunas.
df.rename(
    columns={"VehicleSpeedInstantaneous" : "SpeedInstantaneous", "VehicleSpeedAverage" : "SpeedAverage", 
             "VehicleSpeedVariance" : "SpeedVariance", "VehicleSpeedVariation" : "SpeedVariation",
             "LongitudinalAcceleration" : "LongAcceleration", "VerticalAcceleration" : "VertAcceleration"},
    inplace=True)

#Remove colunas indesejadas do dataframe.
df.drop(['AltitudeVariation', 'EngineLoad', 'EngineCoolantTemperature', 'ManifoldAbsolutePressure', 'EngineRPM', 'MassAirFlow', 'IntakeAirTemperature', 
        'FuelConsumptionAverage', 'traffic', 'drivingStyle'], axis=1, inplace=True)

#Colunas que contêm valores númericos.
colunasNumericas = ['SpeedInstantaneous', 'SpeedAverage', 'SpeedVariance', 'SpeedVariation', 'LongAcceleration', 'VertAcceleration']

#Substitui ',' por '.' nas colunas 'SpeedInstantaneous', 'SpeedAverage', 'SpeedVariance', 'SpeedVariation', 'LongAcceleration', 'VertAcceleration'
#e altera seu tipo para np.float64.
df = df.replace(',', '.', regex=True)
df[colunasNumericas] = df[colunasNumericas].astype(np.float64)

#Remove linhas com valores não numéricos (NaN).
df.dropna(inplace=True)

#Substitui as linhas que contêm 'fullOfHoles' na coluna 'roadSurface' por 'unevenCondition'.
df['roadSurface'] = df['roadSurface'].replace({"FullOfHolesCondition" : "UnevenCondition"})

#Corrige a numeração do índice, começando em 0 e terminando em 23765.
df.reset_index(drop=True, inplace=True)

#Cria um codificador para associar os dados da coluna 'roadSurface' a dados numéricos.
classes = df['roadSurface'].values.reshape(-1,1)
enc = OneHotEncoder(sparse_output=False)
enc.fit_transform(classes)

#Normaliza os dados das colunas numéricas para que fiquem na mesma escala utilizando a função 'QuantileTranformer'.
scaled_features = QuantileTransformer().fit_transform(df[colunasNumericas].values)
scaled_df = pd.DataFrame(scaled_features, index=df[colunasNumericas].index, columns=df[colunasNumericas].columns)

#Exibe todas as colunas numéricas em forma de um conjunto de histogramas.
#fig = plt.figure(figsize=(15,20))
#ax = fig.gca()
#scaled_df.hist(ax=ax)

#print(df.info())

#plt.show()