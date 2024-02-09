import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

#abre o csv e transforma em um dataframe
corsa_01 = r'opel_corsa_01.csv'
df_corsa_01 = pd.read_csv(corsa_01, sep=';')

corsa_02 = r'opel_corsa_02.csv'
df_corsa_02 = pd.read_csv(corsa_02, sep=';')

peugeot_01 = r'peugeot_207_01.csv'
df_peugeot_01 = pd.read_csv(peugeot_01, sep=';')

peugeot_02 = r'peugeot_207_02.csv'
df_peugeot_02 = pd.read_csv(peugeot_02, sep=';')

#concatena todos os dataframes em um unico
df_concat = pd.concat([df_corsa_01, df_corsa_02, df_peugeot_01, df_peugeot_02], axis=0)

#remove colunas indesejadas do dataframe
df_concat.drop(['AltitudeVariation', 'EngineLoad', 'EngineCoolantTemperature', 'ManifoldAbsolutePressure', 'EngineRPM', 'MassAirFlow', 'IntakeAirTemperature', 
            'FuelConsumptionAverage', 'traffic', 'drivingStyle'], axis=1, inplace=True)

#colunas que contem valores numericos
colunasNumericas = ['VehicleSpeedInstantaneous', 'VehicleSpeedAverage', 'VehicleSpeedVariance', 'VehicleSpeedVariation', 'LongitudinalAcceleration', 'VerticalAcceleration']

#informacoes sobre as colunas do dataframe
print(df_concat.info())

#substitui ',' por '.' nas strings do dataframe e seta tipo para float
df_concat = df_concat.replace(',', '.', regex=True)
df_concat[colunasNumericas] = df_concat[colunasNumericas].astype(np.float64)

#verifica possiveis valores para coluna roadSurface
print(list(df_concat['roadSurface'].unique()))

#remove linhas com valores nao numericos (NaN)
df_concat.dropna(inplace=True)

print(df_concat['VerticalAcceleration'].isnull().values.any())

#X = dados[['VehicleSpeedInstantaneous']].values
#y = dados[['LongitudinalAcceleration']].values
#plt.scatter(X, y, c=y)

plt.show()