import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import QuantileTransformer
import seaborn as sb
import pandas.plotting as pp
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve, roc_auc_score, confusion_matrix, classification_report, accuracy_score

def plot_roc_curve(fpr, tpr):
    '''
    Plota um grafico da curva de ROC dado a quantidade de falsos positivos e de verdadeiros positivos
    '''
    # Plota a curva
    plt.plot(fpr, tpr, color='orange', label='ROC')
    #Plota a baseline
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--', label='Guessing')

    #Customiza o plot
    plt.xlabel('False positive rate (fpr)')
    plt.ylabel('True positive rate (tpr)')
    plt.title('Receiver Operator Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

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

def ROC_curve_validation(clf, X_test):
    '''
    Realiza a validação de um modelo através do método de Receiving Operating Characteristic (ROC Curve).
    Recebe como parâmetros o modelo a ser validado e uma porção de dados X para teste.
    '''
    y_probs = clf.predict_proba(X_test)
    y_probs_positive = y_probs[:, 1]
    # Transforma os valores de y em 0 ou 1.
    y_test_nums = np.array(y_test=='UnevenCondition', dtype=int)
    # Instancia os elementos da ROC Curve.
    fpr, tpr, threshhold = roc_curve(y_test_nums, y_probs_positive)
    # Plota o gráfico.
    plot_roc_curve(fpr, tpr)
    # Faz o score da área embaixo da curva.
    print('Esse é o AUC score da ROC Curve: ', roc_auc_score(y_test_nums, y_probs_positive))

def confusion_matrix_validation(clf, X_test, y_test):
    '''
    Realiza a validação de um modelo através de uma matriz de confusão.
    Recebe como parâmetros o modelo a ser validado, uma porção de dados X para teste e uma porção de dados y para teste.
    '''
    y_preds = clf.predict(X_test)
    conf_mat = confusion_matrix(y_test, y_preds)
    plot_conf_mat(conf_mat)

def accuracy_validation(clf, X_test, y_test, X, y):
    '''
    Realiza a validação de um modelo utilizando a acurácia (accuracy).
    Recebe como parâmetros o modelo a ser validado, uma porção de dados X para teste, uma porção de dados y para teste e o conjunto total X e y de dados.
    '''
    clf_single_score = clf.score(X_test, y_test)
    clf_cross_val_score = np.mean(cross_val_score(clf, X, y, cv=9))
    print('Esse é o score do modelo: ', clf_single_score)
    print('Esse é a média dos scores da validação cruzada do modelo: ', clf_cross_val_score)

def class_repo_validation(clf, X_test, y_test):
    '''
    Realiza a validação do modelo utilizando o método de Classification Report.
    Recebe como parâmetros o modelo a ser validado, uma porção de dados X para teste e uma porção de dados y para teste.
    '''
    y_preds = clf.predict(X_test)
    print(classification_report(y_test, y_preds))

# Abre o csv e transforma em um dataframe.
corsa_01 = r'datasets/opel_corsa_01.csv'
df_corsa_01 = pd.read_csv(corsa_01, sep=';')

corsa_02 = r'datasets/opel_corsa_02.csv'
df_corsa_02 = pd.read_csv(corsa_02, sep=';')

peugeot_01 = r'datasets/peugeot_207_01.csv'
df_peugeot_01 = pd.read_csv(peugeot_01, sep=';')

peugeot_02 = r'datasets/peugeot_207_02.csv'
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
'''
# Exibe todas as colunas numéricas em forma de um conjunto de histogramas.
fig = plt.figure(figsize=(15,20))
ax = fig.gca()
df.hist(ax=ax)
plt.show()
'''
# Normaliza os dados das colunas numéricas para que fiquem na mesma escala utilizando a função 'QuantileTranformer'.
scaled_features = QuantileTransformer().fit_transform(df[colunasNumericas].values)
scaled_df = pd.DataFrame(scaled_features, index=df[colunasNumericas].index, columns=df[colunasNumericas].columns)
scaled_df['roadSurface'] = df['roadSurface']

scaled_df.to_csv('dataframe_antigo.csv', encoding='utf-8', index=False)

'''
# Exibe todas as colunas numéricas em forma de um conjunto de histogramas.
fig = plt.figure(figsize=(15,20))
ax = fig.gca()
scaled_df.hist(ax=ax)
plt.show()
'''
# Criando X e y (features e labels).
X = scaled_df[colunasNumericas]
y = scaled_df['roadSurface']

# Separa os dados para teste e treinamento utilizando 70% para treinamento e 30% para teste.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#n_estimators=100, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', max_depth=100
#n_estimators=500, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', max_depth=200, criterion='entropy'

# Utilizando o RandomForestClassifier.
#clf = ExtraTreesClassifier()
#clf.fit(X_train, y_train)

# Teste de performance modelo 2.
#clf2 = RandomForestClassifier(n_estimators=100, min_samples_split=2, min_samples_leaf=1, 
#                              max_features='sqrt', max_depth=100)
#clf2.fit(X_train, y_train)
#print(clf2.score(X_test, y_test))
#y_preds2 = clf2.predict(X_test)
#clf2_metric = evaluate_preds(y_test, y_preds2)
# Teste de performance modelo 3.
#clf3 = RandomForestClassifier(n_estimators=400, min_samples_split=6, min_samples_leaf=1, 
#                              max_features='log2', max_depth=500)
#clf3.fit(X_train, y_train)
#print(clf3.score(X_test, y_test))

#y_preds3 = clf3.predict(X_test)
#clf3_metric = evaluate_preds(y_test, y_preds2)


# Utiliza o RandomizedSearchCV para testar diferentes Hiper-Parâmetros do modelo.
grid = {'n_estimators': range(100, 1200, 100),
        'criterion': ['gini', 'entropy', 'log_loss'],
        'max_depth': [None, 100, 200, 500, 1000],
        'max_features': ['sqrt', 'log2', None],
        'min_samples_split': [2, 10, 25, 100],
        'min_samples_leaf': [1, 2, 3, 4, 5, 10]}

'''
rs_clf = RandomizedSearchCV(estimator=clf, param_distributions=grid, n_iter=10, cv=5, verbose=2)

rs_clf.fit(X_train, y_train)
print('Best Params:')
print(rs_clf.best_params_)
'''


# Teste comparativo
clf = ExtraTreesClassifier(n_estimators=500, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', max_depth=200, criterion='entropy')
clf.fit(X_train, y_train)

accuracy_validation(clf, X_test, y_test, X, y)

ROC_curve_validation(clf, X_test)

confusion_matrix_validation(clf, X_test, y_test)

class_repo_validation(clf, X_test, y_test)
'''
# Utilizando o ExtraTreesClassifier.
clf = ExtraTreesClassifier(n_estimators=500, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', max_depth=200, criterion='entropy')
clf.fit(X_train, y_train)

accuracy_validation(clf, X_test, y_test, X, y)

ROC_curve_validation(clf, X_test)

confusion_matrix_validation(clf, X_test, y_test)

class_repo_validation(clf, X_test, y_test)
'''