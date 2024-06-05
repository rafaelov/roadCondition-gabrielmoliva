import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import seaborn as sb
from sklearn.metrics import confusion_matrix, classification_report

def plot_conf_mat(conf_mat):
    '''
    Plota uma matriz de confusão utilizando o heatmap() da biblioteca seaborn
    '''
    fig, ax = plt.subplots(figsize=(5,5))
    ax = sb.heatmap(conf_mat,
                    annot=True, # Adiciona anotações as caixas da matriz.
                    cbar=False,
                    fmt='g')
    # Nomeia os eixos X e y.
    plt.xlabel('True Label')
    plt.ylabel('Predicted Label')
    # Plota a matriz de confusão.
    plt.show()

def confusion_matrix_validation(clf, X_test, y_test):
    '''
    Realiza a validação de um modelo através de uma matriz de confusão.
    Recebe como parâmetros o modelo a ser validado, uma porção de dados X para teste e uma porção de dados y para teste.
    '''
    y_preds = clf.predict(X_test)
    conf_mat = confusion_matrix(y_test, y_preds)
    plot_conf_mat(conf_mat)

def class_repo_validation(clf, X_test, y_test):
    '''
    Realiza a validação do modelo utilizando o método de Classification Report.
    Recebe como parâmetros o modelo a ser validado, uma porção de dados X para teste e uma porção de dados y para teste.
    '''
    y_preds = clf.predict(X_test)
    print(classification_report(y_test, y_preds))

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
                    'temp_below_suspension', 'latitude', 'longitude', 'regular_road_left', 'bad_road_left']
df = df.drop(colunas_dropadas, axis=1)

# Renomeia colunas
df = df.rename(columns={'good_road_left' : 'good_road', 'acc_x_dashboard' : 'acc_x', 'acc_y_dashboard' : 'acc_y', 'acc_z_dashboard' : 'acc_z', 'gyro_x_dashboard' : 'gyro_x',
                        'gyro_y_dashboard' : 'gyro_y', 'gyro_z_dashboard' : 'gyro_z', 'good_road_left' : 'good_road'})

# Numera corretamente os indices do dataframe
df = df.reset_index()

# Aplicando a janela deslizante
colunas_janela = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
tamanho = 100
janela = df[colunas_janela].rolling(window=tamanho)
df_janelado = janela.std().add_suffix('_std')
df_janelado = df_janelado.join(df['speed'], how='right')
df_janelado = df_janelado.join(df['good_road'], how='right')
df_janelado = df_janelado.dropna()
df_janelado = df_janelado.reset_index()


# Criando X e y 
X = df_janelado.drop(['good_road'], axis=1)
y = df_janelado['good_road']

# Separando X e y em teste e treino
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
param = {
    'n_estimators' : [175, 200, 225, 250],
    'criterion' : ['gini', 'entropy'],
    'max_depth' : [None, 50, 100, 200, 250],
    'max_features': ['sqrt'],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [5, 10, 15]
}
'''
# Utilizando o RandomizedSearchCV para encontrar os melhores hiperparametros
# Melhores parametros: {'n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 10, 'max_features': 'sqrt', 'max_depth': None, 'criterion': 'gini'}
# Melhores parametros2:{'n_estimators': 200, 'min_samples_split': 10, 'min_samples_leaf': 10, 'max_features': 'sqrt', 'max_depth': 250, 'criterion': 'entropy'}
clf = ExtraTreesClassifier()
rs_clf = RandomizedSearchCV(estimator=clf, param_distributions=param, n_iter=50, cv=5, verbose=2)
rs_clf.fit(X, y)

print('Melhor score: ', rs_clf.best_score_)
print('Melhores parametros: ', rs_clf.best_params_)
'''


# Utilizando o GridSearchCV
clf = ExtraTreesClassifier()
gridSearch = GridSearchCV(estimator=clf, param_grid=param, cv=5, n_jobs=-1, verbose=2)
gridSearch.fit(X, y)

print('Melhor score: ', gridSearch.best_score_)
print('Melhores parametros: ', gridSearch.best_params_)

'''
# Importando e treinando o classificador
clf = ExtraTreesClassifier(n_estimators=200, min_samples_split=2, min_samples_leaf=10, max_features='sqrt', max_depth=None, criterion='gini')
clf = ExtraTreesClassifier()
clf.fit(X_train, y_train)

score = clf.score(X_test, y_test)
cv_score = np.mean(cross_val_score(clf, X, y, cv=5))
print(f'Score janelamento de dp, window={tamanho}, ExtraTrees: {score}')
print(f'CV janelamento de dp, window={tamanho}, ExtraTrees: {cv_score}')
class_repo_validation(clf, X_test, y_test)
confusion_matrix_validation(clf, X_test, y_test)
'''