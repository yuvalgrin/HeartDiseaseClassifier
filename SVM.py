import numpy as np
import torch
import xgboost as xgb
from torch import optim
from torch.autograd import Variable
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm

from CsvLoader import CsvLoader


def main(data_set):
    seed = 7
    test_size = 0.1
    X_train, X_test, y_train, y_test = train_test_split(data_set.x_train, data_set.y_train,
                                                        test_size=test_size, random_state=seed)
    X_train = np.nan_to_num(X_train)
    X_test = np.nan_to_num(X_test)

    model = svm.SVC(kernel='linear', C=1.0)  # Our model
    model.fit(X_train, y_train)

    # make predictions for test data
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    #
    # # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
