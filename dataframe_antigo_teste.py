import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
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

df = pd.read_csv('dataframe_antigo.csv')

X = df.drop(['roadSurface'], axis=1)
y = df['roadSurface']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

clf = ExtraTreesClassifier()
clf.fit(X_train, y_train)

accuracy_validation(clf, X_test, y_test, X, y)

ROC_curve_validation(clf, X_test)

confusion_matrix_validation(clf, X_test, y_test)

class_repo_validation(clf, X_test, y_test)
