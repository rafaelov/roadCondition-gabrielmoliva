import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import QuantileTransformer, OneHotEncoder
from sklearn.neighbors import KNeighborsRegressor
import seaborn as sb
import pandas.plotting as pp
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

# Abre o csv e transforma em um dataframe.
corsa_01 = r'opel_corsa_01.csv'
df_corsa_01 = pd.read_csv(corsa_01, sep=';')

corsa_02 = r'opel_corsa_02.csv'
df_corsa_02 = pd.read_csv(corsa_02, sep=';')

peugeot_01 = r'peugeot_207_01.csv'
df_peugeot_01 = pd.read_csv(peugeot_01, sep=';')

peugeot_02 = r'peugeot_207_02.csv'
df_peugeot_02 = pd.read_csv(peugeot_02, sep=';')

# Concatena todos os dataframes em um único.
df = pd.concat([df_corsa_01, df_corsa_02, df_peugeot_01, df_peugeot_02], axis=0)

# Renomea colunas.
df.rename(
    columns={"VehicleSpeedInstantaneous" : "SpeedInstantaneous", "VehicleSpeedAverage" : "SpeedAverage", 
             "VehicleSpeedVariance" : "SpeedVariance", "VehicleSpeedVariation" : "SpeedVariation",
             "LongitudinalAcceleration" : "LongAcceleration", "VerticalAcceleration" : "VertAcceleration"},
    inplace=True)

# Remove colunas indesejadas do dataframe.
df.drop(['AltitudeVariation', 'EngineLoad', 'EngineCoolantTemperature', 'ManifoldAbsolutePressure', 'EngineRPM', 'MassAirFlow', 'IntakeAirTemperature', 
        'FuelConsumptionAverage', 'traffic', 'drivingStyle'], axis=1, inplace=True)

# Colunas que contêm valores númericos.
colunasNumericas = ['SpeedInstantaneous', 'SpeedAverage', 'SpeedVariance', 'SpeedVariation', 'LongAcceleration', 'VertAcceleration']

# Substitui ',' por '.' nas colunas 'SpeedInstantaneous', 'SpeedAverage', 'SpeedVariance', 'SpeedVariation', 'LongAcceleration', 'VertAcceleration'
# e altera seu tipo para np.float64.
df = df.replace(',', '.', regex=True)
df[colunasNumericas] = df[colunasNumericas].astype(np.float64)

# Remove linhas com valores não numéricos (NaN).
df.dropna(inplace=True)

# Substitui as linhas que contêm 'fullOfHoles' na coluna 'roadSurface' por 'unevenCondition'.
df['roadSurface'] = df['roadSurface'].replace({"FullOfHolesCondition" : "UnevenCondition"})

# Corrige a numeração do índice, começando em 0 e terminando em 23765.
df.reset_index(drop=True, inplace=True)

# Exibe todas as colunas numéricas em forma de um conjunto de histogramas.
#fig = plt.figure(figsize=(15,20))
#ax = fig.gca()
#scaled_df.hist(ax=ax)
#plt.show()

# Normaliza os dados das colunas numéricas para que fiquem na mesma escala utilizando a função 'QuantileTranformer'.
scaled_features = QuantileTransformer().fit_transform(df[colunasNumericas].values)
scaled_df = pd.DataFrame(scaled_features, index=df[colunasNumericas].index, columns=df[colunasNumericas].columns)

# Utilizando o LinearSVC Estimator.
# Criando X e y (features e labels).
X = scaled_df[colunasNumericas]
y = df['roadSurface']

# Separa os dados para teste e treinamento utilizando 70% para treinamento e 30% para teste.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Instanciando o classificador.
clf = LinearSVC(dual='auto',max_iter=1000)
clf.fit(X_train, y_train)

# Validação do classificador.
score = clf.score(X_test, y_test)
print('Esse é o score do LinearSVC: ', score)

# Utilizando o RandomForestClassifier.
# Instanciando o classificador.
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Validação do classificador.
score = clf.score(X_test, y_test)
print('Esse é o score do RandomForestClassifier: ', score)