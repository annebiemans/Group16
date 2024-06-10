#%%
#wat we moeten implementeren:
#1. zorgen dat je als X meerdere features toe kan voegen (molecuul naam, aantal ringen, aantal H-atomen, bijv)
#2. zorgen dat je als y meerdere features toe kan voegen (zowel inhibit bij mol1 als inhibit bij mol2).
# dan ook ff zorgen dat je het juiste model gebruikt: model = MultiOutputClassifier(RandomForestClassifier())
import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem import PandasTools

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def read_data():
    data_raw = pd.read_csv('tested_molecules.csv')
    return data_raw

def data_prep_fp(data_raw):
    df_molecules = pd.DataFrame(data_raw['SMILES'])
    PandasTools.AddMoleculeColumnToFrame(data_raw, smilesCol='SMILES')
    #df_molecules['canonical'] = [Chem.MolToSmiles(Chem.MolFromSmiles(x), canonical=True) for x in df_molecules['SMILES']]
    df_molecules['mol'] = [Chem.MolFromSmiles(x) for x in df_molecules['SMILES']]
    df_molecules['fp'] = [AllChem.GetMorganFingerprintAsBitVect(x, 2, nBits=1024) for x in df_molecules['mol']]
    df_molecules['Num_Bonds'] = [x.GetNumBonds() for x in df_molecules['mol']]
    return df_molecules

def machine_learning():
    X = df_molecules['fp'].tolist()
    #X = df_molecules['Num_Bonds'].to_numpy().reshape(-1, 1)        dit werkt dus ook..?
    y = data_raw['ERK2_inhibition']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    
    clf.fit(X_train, y_train)

    return clf, X_test, y_test, X_train, y_train

def predict(clf, X_test, y_test, X_train, y_train):    
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=1)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    print('Accuracy: ', accuracy)
    print('Precision: ', precision)
    print('Recall: ', recall)
    print('F1: ', f1)
    print('ROC AUC: ', roc_auc)

    return y_pred

def visualize():
    #Draw.MolsToGridImage(df_molecules['mol'], molsPerRow=4, subImgSize=(200,200))
    importances = clf.feature_importances_

    return importances

if __name__ == '__main__':
    data_raw = read_data()
    df_molecules = data_prep_fp(data_raw)
    print(df_molecules)
    clf, X_test, y_test, X_train, y_train = machine_learning()
    y_pred = predict(clf, X_test, y_test, X_train, y_train)
    importances = visualize()
    print(importances)



