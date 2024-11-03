import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split, cross_val_score
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
import seaborn as sb
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, multilabel_confusion_matrix, ConfusionMatrixDisplay
from sklearn.neural_network import MLPClassifier
import time

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
    print(classification_report(y_test, y_preds, zero_division=0.0, digits=4))

def plot_multilabel_conf_mat(clf, X_test, y_test):
    '''
    Plota uma matriz de confusão para múltiplas classes.
    Recebe como parâmetros o modelo a ser validado, uma porção de dados X para teste e uma porção de dados y para teste.
    '''
    y_preds = clf.predict(X_test)
    conf_mats = multilabel_confusion_matrix(y_test, y_preds)
    for confusion_matrix in conf_mats:
        disp = ConfusionMatrixDisplay(confusion_matrix, display_labels=['good_road', 'regular_road', 'bad_road'])
        disp.plot(include_values=True, cmap='viridis', ax=None, xticks_rotation='vertical')
        plt.show()

# Abrindo os CSV
df_janelado = pd.read_csv(r'dataframe_janelado_mean_100.csv')
df_janelado = df_janelado.drop(['index'], axis=1)

# Criando X e y
#classes = ['good_road', 'regular_road', 'bad_road']
X = df_janelado.drop(['good_road'], axis=1)
y = df_janelado['good_road']

# Separando X e y em teste e treino
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
'''
# Utiliza o RandomizedSearchCV para testar diferentes Hiper-Parâmetros do modelo.
grid = {'n_estimators': range(100, 1200, 100),
        'criterion': ['gini', 'entropy', 'log_loss'],
        'max_depth': [None, 100, 200, 500, 1000],
        'max_features': ['sqrt', 'log2', None],
        'min_samples_split': [2, 10, 25, 100],
        'min_samples_leaf': [1, 2, 3, 4, 5, 10]}

clf = ExtraTreesClassifier()
rs_clf = RandomizedSearchCV(estimator=clf, param_distributions=grid, n_iter=10, cv=5, verbose=2)

rs_clf.fit(X_train, y_train)
print('Best Params:')
print(rs_clf.best_params_)
#melhores:'n_estimators': 300, 'min_samples_split': 10, 'min_samples_leaf': 2, 'max_features': None, 'max_depth': None, 'criterion': 'gini'
'''
'''
# Utiliza o GridSearchCV para testar diferentes Hiper-Parâmetros do modelo.
grid = {'n_estimators': range(200, 400, 10),
        'criterion': ['gini'],
        'max_depth': [None],
        'max_features': [None],
        'min_samples_split': [5, 10, 15],
        'min_samples_leaf': [1, 2, 3]}

clf = ExtraTreesClassifier()
gs_clf = GridSearchCV(estimator=clf, param_grid=grid, n_jobs=2, verbose=2)

gs_clf.fit(X_train, y_train)
print('Best Params:')
print(gs_clf.best_params_)
'''
#start = time.time()
# Importando e treinando o classificador
clf = ExtraTreesClassifier()
clf.fit(X_train, y_train)

score = clf.score(X_test, y_test)
#end = time.time()
#print("Tempo de processamento: ", end - start)

cv_score = np.mean(cross_val_score(clf, X, y, cv=5))
print(f'Score janelamento de dp, window=100, ExtraTrees: {score}')
print(f'CV janelamento de dp, window=100, ExtraTrees: {cv_score}')
class_repo_validation(clf, X_test, y_test)
#plot_multilabel_conf_mat(clf, X_test, y_test)
#confusion_matrix_validation(clf, X_test, y_test)
#ROC_curve_validation(clf, X_test, y_test)

'''
# Importando e treinando a rede neural
mlp = MLPClassifier(hidden_layer_sizes=(6,))
mlp.fit(X_train, y_train)

score = mlp.score(X_test, y_test)
cv_score = np.mean(cross_val_score(mlp, X, y, cv=5))
print(f'Score janelamento de dp, window=100, ExtraTrees: {score}')
print(f'CV janelamento de dp, window=100, ExtraTrees: {cv_score}')
class_repo_validation(mlp, X_test, y_test)
confusion_matrix_validation(mlp, X_test, y_test)
ROC_curve_validation(mlp, X_test, y_test)
'''