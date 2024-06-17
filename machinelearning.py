#%%
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, balanced_accuracy_score, confusion_matrix, make_scorer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler

def read_data():
    data_raw = pd.read_csv('tested_molecules.csv')
    df_molecules = pd.read_csv('descriptors.csv')
    df_untested_molecules = pd.read_csv('untested_molecules.csv')
    
    return data_raw, df_molecules, df_untested_molecules

# dit stukje hieronder zorgt dat de score van randomsearch hierop worden gefilterd
def sensitivity_specificity_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return (sensitivity + specificity) / 2

def machine_learning(df_molecules, data_raw, kinase, descriptors_used):
    X = df_molecules.iloc[:, 0:descriptors_used]
    y = data_raw[[kinase]]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rfc = MultiOutputClassifier(RandomForestClassifier(random_state=42))
    knn = MultiOutputClassifier(KNeighborsClassifier())
    gbc = MultiOutputClassifier(GradientBoostingClassifier(random_state=42))

    scorer = make_scorer(sensitivity_specificity_score)

    param_grid_rfc = {
        'estimator__n_estimators': [50, 100, 200, 300],
        'estimator__max_depth': [None, 20, 30, 40, 50],
        'estimator__min_samples_split': [2, 5, 10],
        'estimator__min_samples_leaf': [1, 2, 4, 8, 10],
        'estimator__max_features': ['sqrt', 'log2', 0.2, 0.3, 0.4]
    }

    param_grid_knn = {
        'estimator__n_neighbors':  list(range(10, 30)),
        'estimator__weights': ['uniform', 'distance'],
        'estimator__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'estimator__p': [1, 2]
    }

    param_grid_gbc = {
        'estimator__n_estimators': [50, 100, 200],
        'estimator__learning_rate': [0.01, 0.1, 0.5],
        'estimator__max_depth': [3, 4, 5],
        'estimator__min_samples_split': [2, 5, 10],
        'estimator__min_samples_leaf': [1, 2, 4],
        'estimator__max_features': ['sqrt', 'log2', 0.2, 0.3, 0.4],
        'estimator__subsample': [0.8, 0.9, 1.0]
    }

    n_iter_search = 100
    random_search_rfc = RandomizedSearchCV(estimator=rfc, param_distributions=param_grid_rfc, n_iter=n_iter_search, cv=5, random_state=42, n_jobs=-1, scoring=scorer)
    random_search_rfc.fit(X_train, y_train)
    # print("Best parameters found by RandomizedSearchCV for RandomForestClassifier: ", random_search_rfc.best_params_)
    
    random_search_knn = RandomizedSearchCV(estimator=knn, param_distributions=param_grid_knn, n_iter=n_iter_search, cv=5, random_state=42, n_jobs=-1, scoring=scorer)
    random_search_knn.fit(X_train, y_train)
    # print("Best parameters found by RandomizedSearchCV for KNeighborsClassifier: ", random_search_knn.best_params_)
    
    random_search_gbc = RandomizedSearchCV(estimator=gbc, param_distributions=param_grid_gbc, n_iter=n_iter_search, cv=5, random_state=42, n_jobs=-1, scoring=scorer)
    random_search_gbc.fit(X_train, y_train)
    # print("Best parameters found by RandomizedSearchCV for GradientBoostingClassifier: ", random_search_gbc.best_params_)
    
    best_model_rfc = random_search_rfc.best_estimator_
    best_model_knn = random_search_knn.best_estimator_
    best_model_gbc = random_search_gbc.best_estimator_
    
    return best_model_rfc, best_model_knn, best_model_gbc, X_test, y_test, X_train, y_train

def predicting(clf, X_test, y_test, X_train, y_train):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred_df = pd.DataFrame(y_pred, columns=[kinase])

    for target in [kinase]:
        cm = confusion_matrix(y_test[target], y_pred_df[target])
        tn, fp, fn, tp = cm.ravel()
        accuracy = accuracy_score(y_test[target], y_pred_df[target])
        precision = precision_score(y_test[target], y_pred_df[target], zero_division=1)
        sensitivity = recall_score(y_test[target], y_pred_df[target])
        balanced_accuracy = balanced_accuracy_score(y_test[target], y_pred_df[target])
        
        print(f'\t{target} - Balanced Accuracy: {balanced_accuracy}, Accuracy: {accuracy}, Precision: {precision}, Sensitivity: {sensitivity}')

    return y_pred

if __name__ == '__main__':
    data_raw, df_molecules, df_untested_molecules = read_data()

    for kinase, descriptors_used in [['PKM2_inhibition', 10], ['ERK2_inhibition', 2]]:
        print('\nKINASE:', kinase, "\n")
        
        best_model_rfc, best_model_knn, best_model_gbc, X_test, y_test, X_train, y_train = machine_learning(df_molecules, data_raw, kinase, descriptors_used)

        print("Evaluating RandomForestClassifier")
        y_pred_rfc = predicting(best_model_rfc, X_test, y_test, X_train, y_train)
        # print('predicted:', y_pred_rfc.sum().sum())
        # print('actual:', y_test.sum().sum())

        print("Evaluating KNeighborsClassifier")
        y_pred_knn = predicting(best_model_knn, X_test, y_test, X_train, y_train)
        # print('predicted:', y_pred_knn.sum().sum())
        # print('actual:', y_test.sum().sum())

        print("Evaluating GradientBoostingClassifier")
        y_pred_gbc = predicting(best_model_gbc, X_test, y_test, X_train, y_train)
        # print('predicted:', y_pred_gbc.sum().sum())
        # print('actual:', y_test.sum().sum())

        # based on balanced average (stated on canvas), for PKM2 gbc works best and for ERK2 rfc works best
        if kinase == 'PKM2_inhibition':
            best_model_PKM2 = best_model_gbc
        elif kinase == 'ERK2_inhibition':
            best_model_PKM2 = best_model_rfc

        # train model on complete dataset --> maak hier def van?
        X_full = df_molecules.iloc[:, 0:descriptors_used]
        y_full = data_raw[[kinase]]

        scaler = StandardScaler()
        X_full = scaler.fit_transform(X_full)

        if kinase == 'PKM2_inhibition':
            best_model_PKM2.fit(X_full, y_full)
            print('scores after training gbc on full dataset')
            y_pred_PKR2 = predicting(best_model_PKM2, X_full, y_full, X_full, y_full)
        elif kinase == 'ERK2_inhibition':
            best_model_ERK2.fit(X_full, y_full)
            print('scores after training rfc on full dataset')
            y_pred_ERK2 = predicting(best_model_ERK2, X_full, y_full, X_full, y_full)

# nog toevoegen dat hij inhibition voor untested molecules berekent