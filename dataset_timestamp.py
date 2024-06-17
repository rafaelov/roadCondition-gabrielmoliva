import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import seaborn as sb
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve

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

def ROC_curve_validation(clf, X_test, y_test):
    '''
    Realiza a validação de um modelo através do método de Receiving Operating Characteristic (ROC Curve).
    Recebe como parâmetros o modelo a ser validado e uma porção de dados X para teste.
    '''
    y_probs = clf.predict_proba(X_test)
    y_probs_positive = y_probs[:, 1]
    # Instancia os elementos da ROC Curve.
    fpr, tpr, threshhold = roc_curve(y_test, y_probs_positive)
    # Plota o gráfico.
    plot_roc_curve(fpr, tpr)
    # Faz o score da área embaixo da curva.
    print('Esse é o AUC score da ROC Curve: ', roc_auc_score(y_test, y_probs_positive))

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

# Abrindo os CSV
df_janelado = pd.read_csv(r'dataframe_janelado.csv')

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

'''
# Utilizando o GridSearchCV
# Melhores parametros:  {'criterion': 'entropy', 'max_depth': 250, 'max_features': 'sqrt', 'min_samples_leaf': 15, 'min_samples_split': 10, 'n_estimators': 200}
clf = ExtraTreesClassifier()
gridSearch = GridSearchCV(estimator=clf, param_grid=param, cv=5, n_jobs=-1, verbose=2)
gridSearch.fit(X, y)

print('Melhor score: ', gridSearch.best_score_)
print('Melhores parametros: ', gridSearch.best_params_)
'''

# Importando e treinando o classificador
#clf = ExtraTreesClassifier(criterion='entropy', max_depth=250, max_features='sqrt', min_samples_leaf=15, min_samples_split=10, n_estimators=200)
clf = ExtraTreesClassifier()
clf.fit(X_train, y_train)

score = clf.score(X_test, y_test)
cv_score = np.mean(cross_val_score(clf, X, y, cv=5))
print(f'Score janelamento de dp, window=100, ExtraTrees: {score}')
print(f'CV janelamento de dp, window=100, ExtraTrees: {cv_score}')
class_repo_validation(clf, X_test, y_test)
confusion_matrix_validation(clf, X_test, y_test)
ROC_curve_validation(clf, X_test, y_test)