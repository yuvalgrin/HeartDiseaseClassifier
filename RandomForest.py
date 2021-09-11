import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from CsvLoader import CsvLoader

# Validate the
def cross_validator(X_train, y_train, X_test, y_test):
    best_depth = 0
    best_estimator = 0
    best_acc = float("Inf")
    for estimator in range(0, 20):
        for depth in range(0, 20):
            model = RandomForestClassifier(max_depth=8, n_estimators=20)
            model.fit(X_train, y_train)
            mean = 0
            for i in range(4):
                y_pred = model.predict(X_test)
                predictions = [round(value) for value in y_pred]

                # evaluate predictions
                mean += accuracy_score(y_test, predictions)
            mean = mean / 4
            print("Accuracy: %.2f%%" % (mean * 100.0))

            # Update best score
            if mean < best_acc:
                best_acc = mean
                best_depth = depth
                best_estimator = estimator
                print("Best depth: {}, estimator: {}, accuracy: {}".format(best_depth, best_estimator, best_acc))

    print("Best depth: {}, estimator: {}, accuracy: {}".format(best_depth, best_estimator, best_acc))


def main(data_set):
    # split data into train and test sets
    seed = 7
    test_size = 0.2
    X_train, X_test, y_train, y_test = train_test_split(data_set.x_train, data_set.y_train,
                                                        test_size=test_size, random_state=seed)
    X_train = np.nan_to_num(X_train)
    X_test = np.nan_to_num(X_test)

    # Create and train the model
    model = RandomForestClassifier(max_depth=24, n_estimators=25)
    model.fit(X_train, y_train)

    # Use cross validation to find the best Hyper-parameters
    # cross_validator(X_train, y_train, X_test, y_test)

    # Make predictions for test data
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]

    # Evaluate the model predictions
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
