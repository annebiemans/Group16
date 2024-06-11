#%%

import pandas as pd
#import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm


training_csv = 'tested_molecules'

data = pd.read_csv(training_csv + ".csv", usecols=[0])
data_labels = pd.read_csv(training_csv + ".csv", usecols=[1, 2])

X_train, X_test, y_train, y_test = train_test_split(
    data, data_labels, test_size=0.4, random_state=0)



