#%%
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, PandasTools
from rdkit.Chem import rdMolDescriptors
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, balanced_accuracy_score, confusion_matrix
from sklearn.multioutput import MultiOutputClassifier

def read_data():
    data_raw = pd.read_csv('tested_molecules.csv')
    return data_raw

def data_prep_fp(data_raw):
    df_molecules = pd.DataFrame(data_raw['SMILES'])
    PandasTools.AddMoleculeColumnToFrame(df_molecules, smilesCol='SMILES')
    
    df_molecules['mol'] = [Chem.MolFromSmiles(x) for x in df_molecules['SMILES']]
    df_molecules['Num_H_Donors'] = df_molecules['mol'].apply(lambda x: rdMolDescriptors.CalcNumHBD(x))
    df_molecules['LogP'] = df_molecules['mol'].apply(lambda x: rdMolDescriptors.CalcCrippenDescriptors(x)[0])
    df_molecules['Num_Rings'] = df_molecules['mol'].apply(lambda x: rdMolDescriptors.CalcNumRings(x))
    df_molecules['Num_H_Acceptors'] = df_molecules['mol'].apply(lambda x: rdMolDescriptors.CalcNumHBA(x))
    
    return df_molecules

def machine_learning(df_molecules, data_raw):
    X = df_molecules[['Num_H_Donors', 'LogP', 'Num_Rings', 'Num_H_Acceptors']]
    y = data_raw[['PKM2_inhibition', 'ERK2_inhibition']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rfc = MultiOutputClassifier(RandomForestClassifier(random_state=42))
    knn = MultiOutputClassifier(KNeighborsClassifier())

    param_grid_rfc = {
        'estimator__n_estimators': [50, 100, 200],
        'estimator__max_depth': [None, 10, 20, 30],
        'estimator__min_samples_split': [2, 5, 10],
        'estimator__min_samples_leaf': [1, 2, 4],
        'estimator__max_features': ['sqrt', 'log2']
    }

    param_grid_knn = {
        'estimator__n_neighbors': list(range(1, 31)),
        'estimator__weights': ['uniform', 'distance'],
        'estimator__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'estimator__p': [1, 2]
    }

    n_iter_search = 100
    random_search_rfc = RandomizedSearchCV(estimator=rfc, param_distributions=param_grid_rfc, n_iter=n_iter_search, cv=5, random_state=42, n_jobs=-1, verbose=2)
    random_search_rfc.fit(X_train, y_train)
    print("Best parameters found by RandomizedSearchCV for RandomForestClassifier: ", random_search_rfc.best_params_)
    
    random_search_knn = RandomizedSearchCV(estimator=knn, param_distributions=param_grid_knn, n_iter=n_iter_search, cv=5, random_state=42, n_jobs=-1, verbose=2)
    random_search_knn.fit(X_train, y_train)
    print("Best parameters found by RandomizedSearchCV for KNeighborsClassifier: ", random_search_knn.best_params_)
    
    best_model_rfc = random_search_rfc.best_estimator_
    best_model_knn = random_search_knn.best_estimator_
    
    return best_model_rfc, best_model_knn, X_test, y_test, X_train, y_train

def predicting(clf, X_test, y_test, X_train, y_train):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred_df = pd.DataFrame(y_pred, columns=['PKM2_inhibition', 'ERK2_inhibition'])

    for target in ['PKM2_inhibition', 'ERK2_inhibition']:
        cm = confusion_matrix(y_test[target], y_pred_df[target])
        tn, fp, fn, tp = cm.ravel()
        accuracy = accuracy_score(y_test[target], y_pred_df[target])
        precision = precision_score(y_test[target], y_pred_df[target], zero_division='warn')
        sensitivity = recall_score(y_test[target], y_pred_df[target])
        balanced_accuracy = balanced_accuracy_score(y_test[target], y_pred_df[target])
        
        print(f'{target} - Accuracy: {accuracy}, Precision: {precision}, Sensitivity: {sensitivity}, Balanced Accuracy: {balanced_accuracy}')

    return y_pred

def visualize(model):
    importances = model.estimators_[0].feature_importances_
    feature_names = ['Num_H_Donors', 'LogP', 'Num_Rings', 'Num_H_Acceptors']
    feature_importances = pd.DataFrame(importances, index=feature_names, columns=['importance']).sort_values('importance', ascending=False)
    print(feature_importances)

if __name__ == '__main__':
    data_raw = read_data()
    df_molecules = data_prep_fp(data_raw)
    best_model_rfc, best_model_knn, X_test, y_test, X_train, y_train = machine_learning(df_molecules, data_raw)
    
    print("Evaluating RandomForestClassifier")
    y_pred_rfc = predicting(best_model_rfc, X_test, y_test, X_train, y_train)
    
    print("Evaluating KNeighborsClassifier")
    y_pred_knn = predicting(best_model_knn, X_test, y_test, X_train, y_train)
    print(y_pred_knn)
    
    print("Feature Importances for RandomForestClassifier")
    visualize(best_model_rfc)



