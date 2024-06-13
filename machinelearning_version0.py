#%%
#wat we moeten implementeren:
#1. zorgen dat je als X meerdere features toe kan voegen (molecuul naam, aantal ringen, aantal H-atomen, bijv)
#2. zorgen dat je als y meerdere features toe kan voegen (zowel inhibit bij mol1 als inhibit bij mol2).
# dan ook ff zorgen dat je het juiste model gebruikt: model = MultiOutputClassifier(RandomForestClassifier())
import pandas as pd
import rdkit
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem import PandasTools
import math

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.multioutput import MultiOutputClassifier
def read_data():
    data_raw = pd.read_csv('tested_molecules.csv')
    return data_raw

def data_prep_fp(data_raw):
    df_molecules = pd.DataFrame(data_raw['SMILES'])
    PandasTools.AddMoleculeColumnToFrame(data_raw, smilesCol='SMILES')
    #hier dus de goede descriptors invoegen, moeten numerieke waarde hebben
    df_molecules['mol'] = [Chem.MolFromSmiles(x) for x in df_molecules['SMILES']]
    df_molecules['fp'] = [AllChem.GetMorganFingerprintAsBitVect(x, 2, nBits=1024) for x in df_molecules['mol']]
    df_molecules['Num_H_Donors'] = df_molecules['mol'].apply(lambda x: Chem.rdMolDescriptors.CalcNumHBD(x))
    df_molecules['LogP'] = df_molecules['mol'].apply(lambda x: Chem.rdMolDescriptors.CalcCrippenDescriptors(x)[0])
    return df_molecules

def machine_learning():
    #deze aanpassen op basis van descriptors
    X = df_molecules[['Num_H_Donors','LogP']]
    y = data_raw[['PKM2_inhibition','ERK2_inhibition']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # MultiOutputClassifier met RandomForestClassifier
    rfc = MultiOutputClassifier(RandomForestClassifier(random_state=42))

    # MultiOutputClassifier met KNeighborsClassifier
    knn = MultiOutputClassifier(KNeighborsClassifier())

    # Hyperparameter ruimte
    param_grid_rfc = {
        'estimator__n_estimators': [50, 100, 200],
        'estimator__max_depth': [None, 10, 20, 30],
        'estimator__min_samples_split': [2, 5, 10],
        'estimator__min_samples_leaf': [1, 2, 4],
        'estimator__max_features': ['sqrt', 'log2']
    }

    # Hyperparameter ruimte voor KNN
    param_grid_knn = {
        'estimator__n_neighbors': list(range(1, 31)),
        'estimator__weights': ['uniform', 'distance'],
        'estimator__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'estimator__p': [1, 2]
    }

    # Gebruik RandomizedSearchCV om de beste hyperparameters te vinden
    n_iter_search = 100
    random_search = RandomizedSearchCV(estimator=rfc, param_distributions=param_grid_rfc, n_iter=n_iter_search, cv=5, random_state=42, n_jobs=-1, verbose=2)

    random_search.fit(X_train, y_train)
    print("Best parameters found by RandomizedSearchCV: ", random_search.best_params_)
    
    # Haal het beste model op
    best_model = random_search.best_estimator_
    
    # Voorspel op de testset
    y_pred = best_model.predict(X_test)
    
    # Evalueer de prestaties
    print("Classification Report:\n", classification_report(y_test, y_pred))
        

    # Gebruik RandomizedSearchCV om de beste hyperparameters voor KNN te vinden
    random_search_knn = RandomizedSearchCV(estimator=knn, param_distributions=param_grid_knn, n_iter=n_iter_search, cv=5, random_state=42, n_jobs=-1, verbose=2, error_score='raise')

    random_search_knn.fit(X_train, y_train)
    print("Best parameters found by RandomizedSearchCV for KNN: ", random_search_knn.best_params_)
    
    # Haal het beste model op
    best_model_knn = random_search_knn.best_estimator_
    
    # Voorspel op de testset
    y_pred_knn = best_model_knn.predict(X_test)
    
    # Evalueer de prestaties
    print("Classification Report for KNN:\n", classification_report(y_test, y_pred_knn))


    return rfc, knn, X_test, y_test, X_train, y_train

def predict(clf, X_test, y_test, X_train, y_train):
    # we moeten alles los bekijken, want dan krijg je veel beter inzicht.    
    clf.fit(X_train, y_train)
    y_true = y_test
    y_pred = clf.predict(X_test)
    y_pred_df = pd.DataFrame(y_pred, columns =['PKM2_inhibition','ERK2_inhibition'])
    #confusion matrix berekenen
    cm_PKM2 = confusion_matrix(y_true['PKM2_inhibition'],y_pred_df['PKM2_inhibition'])
    tn_PKM2, fp_PKM2, fn_PKM2, tp_PKM2 = cm_PKM2.ravel()

    cm_ERK2 = confusion_matrix(y_true['ERK2_inhibition'],y_pred_df['ERK2_inhibition'])
    tn_ERK2, fp_ERK2, fn_ERK2, tp_ERK2 = cm_ERK2.ravel()
    #accuracy berekenen
    accuracy_PKM2 = accuracy_score(y_true['PKM2_inhibition'], y_pred_df['PKM2_inhibition'])
    accuracy_ERK2 = accuracy_score(y_true['ERK2_inhibition'], y_pred_df['ERK2_inhibition'])
    #precision berekenen
    precision_PKM2 = precision_score(y_true['PKM2_inhibition'], y_pred_df['PKM2_inhibition'], zero_division='warn')
    precision_ERK2 = precision_score(y_true['ERK2_inhibition'], y_pred_df['ERK2_inhibition'], zero_division='warn')
    #sensitivity berekenen
    sensitivity_PMK2 = recall_score(y_true['PKM2_inhibition'], y_pred_df['PKM2_inhibition'])
    sensitivity_ERK2 = recall_score(y_true['ERK2_inhibition'], y_pred_df['ERK2_inhibition'])
    #f1 = f1_score(y_true, y_pred,average = 'macro' ) deze mag eigenlijk echt weg
    #roc_auc = roc_auc_score(y_true, y_pred, average = 'macro') deze kan ook echt weg
    #tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    balanced_accuracy_PKM2 = balanced_accuracy_score(y_true['PKM2_inhibition'], y_pred_df['PKM2_inhibition'])
    balanced_accuracy_ERK2 = balanced_accuracy_score(y_true['ERK2_inhibition'], y_pred_df['ERK2_inhibition'])

    print('Accuracy: ', accuracy_PKM2, accuracy_ERK2)
    print('Precision: ', precision_PKM2, precision_ERK2)
    print('Sensitivity: ', sensitivity_PMK2, sensitivity_ERK2)
    print('Balanced Accuracy; ', balanced_accuracy_PKM2, balanced_accuracy_ERK2)
    #print('F1: ', f1)
    #print('ROC AUC: ', roc_auc)
    #print(tn)

    return y_pred

def visualize():
    #Draw.MolsToGridImage(df_molecules['mol'], molsPerRow=4, subImgSize=(200,200))
    importances = rfc.feature_importances_

    return importances

if __name__ == '__main__':
    data_raw = read_data()
    df_molecules = data_prep_fp(data_raw)
    rfc, knn, X_test, y_test, X_train, y_train = machine_learning()
    clf = knn, rfc
    clf_method = 'knn','rfc'
    method=0
    for i in clf:
        print('method used: ',clf_method[method])
        method+=1
        y_pred = predict(i, X_test, y_test, X_train, y_train)
    #importances = visualize()
    #print(importances)