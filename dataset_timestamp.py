import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Abrindo os CSV em dataframes
mpu_01 = r'datasets/dataset_gps_mpu_left_01.csv'
df_mpu_01 = pd.read_csv(mpu_01)

mpu_02 = r'datasets/dataset_gps_mpu_left_02.csv'
df_mpu_02 = pd.read_csv(mpu_02)

mpu_03 = r'datasets/dataset_gps_mpu_left_03.csv'
df_mpu_03 = pd.read_csv(mpu_03)

mpu_04 = r'datasets/dataset_gps_mpu_left_04.csv'
df_mpu_04 = pd.read_csv(mpu_04)

mpu_05 = r'datasets/dataset_gps_mpu_left_05.csv'
df_mpu_05 = pd.read_csv(mpu_05)

mpu_06 = r'datasets/dataset_gps_mpu_left_06.csv'
df_mpu_06 = pd.read_csv(mpu_06)

mpu_07 = r'datasets/dataset_gps_mpu_left_07.csv'
df_mpu_07 = pd.read_csv(mpu_07)

mpu_08 = r'datasets/dataset_gps_mpu_left_08.csv'
df_mpu_08 = pd.read_csv(mpu_08)

mpu_09 = r'datasets/dataset_gps_mpu_left_09.csv'
df_mpu_09 = pd.read_csv(mpu_09)

lbl_01 = r'datasets/dataset_labels_01.csv'
df_lbl_01 = pd.read_csv(lbl_01)

lbl_02 = r'datasets/dataset_labels_02.csv'
df_lbl_02 = pd.read_csv(lbl_02)

lbl_03 = r'datasets/dataset_labels_03.csv'
df_lbl_03 = pd.read_csv(lbl_03)

lbl_04 = r'datasets/dataset_labels_04.csv'
df_lbl_04 = pd.read_csv(lbl_04)

lbl_05 = r'datasets/dataset_labels_05.csv'
df_lbl_05 = pd.read_csv(lbl_05)

lbl_06 = r'datasets/dataset_labels_06.csv'
df_lbl_06 = pd.read_csv(lbl_06)

lbl_07 = r'datasets/dataset_labels_07.csv'
df_lbl_07 = pd.read_csv(lbl_07)

lbl_08 = r'datasets/dataset_labels_08.csv'
df_lbl_08 = pd.read_csv(lbl_08)

lbl_09 = r'datasets/dataset_labels_09.csv'
df_lbl_09 = pd.read_csv(lbl_09)

# Concatena os datasets com suas respectivas labels, axis=1 para concatenar horizontalmente
df_concat_01 = pd.concat([df_mpu_01, df_lbl_01], axis=1)
df_concat_02 = pd.concat([df_mpu_02, df_lbl_02], axis=1)
df_concat_03 = pd.concat([df_mpu_03, df_lbl_03], axis=1)
df_concat_04 = pd.concat([df_mpu_04, df_lbl_04], axis=1)
df_concat_05 = pd.concat([df_mpu_05, df_lbl_05], axis=1)
df_concat_06 = pd.concat([df_mpu_06, df_lbl_06], axis=1)
df_concat_07 = pd.concat([df_mpu_07, df_lbl_07], axis=1)
df_concat_08 = pd.concat([df_mpu_08, df_lbl_08], axis=1)
df_concat_09 = pd.concat([df_mpu_09, df_lbl_09], axis=1)

# Concatena todos os dataframes em um unico, axis=0 para concatenar verticalmente
df = pd.concat([df_concat_01, df_concat_04, df_concat_09], axis=0)

# Remove colunas indesejadas do dataframe
colunas_dropadas = ['paved_road', 'unpaved_road', 'dirt_road', 'cobblestone_road',  'asphalt_road', 'no_speed_bump', 'speed_bump_asphalt', 'speed_bump_cobblestone', 
                    'good_road_right', 'regular_road_right', 'bad_road_right', 'acc_x_above_suspension', 'acc_y_above_suspension', 'acc_z_above_suspension', 
                    'acc_x_below_suspension', 'acc_y_below_suspension', 'acc_z_below_suspension', 'gyro_x_above_suspension', 'gyro_y_above_suspension', 
                    'gyro_z_above_suspension', 'gyro_x_below_suspension', 'gyro_y_below_suspension', 'gyro_z_below_suspension', 'mag_x_dashboard', 'mag_y_dashboard', 
                    'mag_z_dashboard', 'mag_x_above_suspension', 'mag_y_above_suspension', 'mag_z_above_suspension', 'temp_dashboard', 'temp_above_suspension', 
                    'temp_below_suspension', 'latitude', 'longitude']
df = df.drop(colunas_dropadas, axis=1)

# Une as colunas 'regular_road_left' e 'bad_road_left' em uma única coluna chamada 'bad_road' utilizando a operação booleana 'OR' e remove as colunas que foram unidas
new_column_values = list(df['regular_road_left']|df['bad_road_left'])
df.insert(len(df.axes[1]), 'bad_road', new_column_values)
df = df.drop(['regular_road_left', 'bad_road_left'], axis=1)

# Renomeia colunas
df = df.rename(columns={'good_road_left' : 'good_road', 'acc_x_dashboard' : 'acc_x', 'acc_y_dashboard' : 'acc_y', 'acc_z_dashboard' : 'acc_z', 'gyro_x_dashboard' : 'gyro_x',
                        'gyro_y_dashboard' : 'gyro_y', 'gyro_z_dashboard' : 'gyro_z'})

# Numera corretamente os indices do dataframe
df = df.reset_index()

#df_janelado = pd.DataFrame(columns=['acc_x_median', 'acc_x_std', 'acc_y_median', 'acc_y_std', 'acc_z_median', 'acc_z_std', 'gyro_x_median', 'gyro_x_std', 'gyro_y_median', 
#                                    'gyro_y_std', 'gyro_z_median', 'gyro_z_std', 'speed_median', 'speed_std'])
df_janelado = pd.DataFrame()
'''
# Criando X e y 
X = df.drop(columns=['timestamp', 'timestamp_gps', 'good_road', 'bad_road'])
y = df[['good_road', 'bad_road']]

# Separando X e y em teste e treino
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Importando e treinando o classificador
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

#clf.predict(X_test, y_test)

print(clf.score(X_test, y_test))
print(np.mean(cross_val_score(clf, X, y, cv=5)))
'''
# Aplicando a janela deslizante
tamanho = 10
df_janelado['acc_x_median'] = df['acc_x'].rolling(window=tamanho).median()
#df_janelado['acc_x_std'] = df['acc_x'].rolling(window=1).std()
df_janelado['acc_y_median'] = df['acc_y'].rolling(window=tamanho).median()
#df_janelado['acc_y_std'] = df['acc_y'].rolling(window=1).std()
df_janelado['acc_z_median'] = df['acc_z'].rolling(window=tamanho).median()
#df_janelado['acc_z_std'] = df['acc_z'].rolling(window=1).std()

df_janelado['gyro_x_median'] = df['gyro_x'].rolling(window=tamanho).median()
#df_janelado['gyro_x_std'] = df['gyro_x'].rolling(window=1).std()
df_janelado['gyro_y_median'] = df['gyro_y'].rolling(window=tamanho).median()
#df_janelado['gyro_y_std'] = df['gyro_y'].rolling(window=1).std()
df_janelado['gyro_z_median'] = df['gyro_z'].rolling(window=tamanho).median()
#df_janelado['gyro_z_std'] = df['gyro_z'].rolling(window=1).std()

df_janelado['speed_median'] = df['speed'].rolling(window=tamanho).median()
#df_janelado['speed_std'] = df['speed'].rolling(window=1).std()

print(df_janelado.info())

# Criando X e y 
X = df_janelado
y = df[['good_road', 'bad_road']]

# Separando X e y em teste e treino
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Importando e treinando o classificador
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

#clf.predict(X_test, y_test)

print('Score janelamento de mediana, window=' + tamanho + ': ', clf.score(X_test, y_test))
print('CV janelamento de mediana, window=' + tamanho + ': ', np.mean(cross_val_score(clf, X, y, cv=5)))