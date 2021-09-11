import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm

from CsvLoader import CsvLoader


def main(data_set):
    # Split data into train and test sets
    seed = 7
    test_size = 0.1
    X_train, X_test, y_train, y_test = train_test_split(data_set.x_train, data_set.y_train,
                                                        test_size=test_size, random_state=seed)
    X_train = np.nan_to_num(X_train)
    X_test = np.nan_to_num(X_test)

    # Create and train the model
    model = svm.SVC(kernel='linear', C=1.0)
    model.fit(X_train, y_train)

    # Make predictions for test data
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]

    # Evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    print("SVM accuracy: %.2f%%" % (accuracy * 100.0))
