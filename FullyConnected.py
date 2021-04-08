import torch
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from CsvLoader import CsvLoader


def calculate_accuracy(y_pred, y):
    """ Find the accuracy rates """
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc


def evaluate(model, iterator, criterion, device):
    """ Evaluate values for validation """
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for (samples, labels) in iterator:
            samples = samples.to(device)
            labels = labels.to(device)
            y_pred = model(samples)
            loss = criterion(y_pred, labels)
            acc = calculate_accuracy(y_pred, labels)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def train(model, iterator, optimizer, criterion, device):
    """ Train the model """
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for (samples, labels) in iterator:
        samples = samples.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        y_pred = model(samples)
        loss = criterion(y_pred, labels)
        acc = calculate_accuracy(y_pred, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def cross_validator(params, dtrain, num_boost_round):
    # This can take some time…
    min_mae = float("Inf")
    best_params = None
    for eta in [.3, .2, .1, .05, .01, .005]:
        print("CV with eta={}".format(eta))
        # We update our parameters
        params['eta'] = eta
        # Run and time CV
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            seed=42,
            nfold=5,
            metrics=['mae'],
            early_stopping_rounds=10
        )
        # Update best score
        mean_mae = cv_results['test-mae-mean'].min()
        boost_rounds = cv_results['test-mae-mean'].argmin()
        print("\tMAE {} for {} rounds\n".format(mean_mae, boost_rounds))
        if mean_mae < min_mae:
            min_mae = mean_mae
            best_params = eta
    print("Best params: {}, MAE: {}".format(best_params, min_mae))

def main(data_set):
    # split data into train and test sets
    seed = 7
    test_size = 0.2
    X_train, X_test, y_train, y_test = train_test_split(train_data_set.x_train, train_data_set.y_train, test_size=test_size, random_state=seed)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    num_boost_round = 500
    params = {
                 # Parameters that we are going to tune.
                 'max_depth': 6,
                 'min_child_weight': 0,
                 'eta': .003,
                 'subsample': 1,
                 'colsample_bytree': 1,
                 # Other parameters
                 'objective': 'reg:linear',
             }
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtest, "Test")],
        early_stopping_rounds=10
    )
    print("Best MAE: {:.2f} in {} rounds".format(model.best_score, model.best_iteration + 1))

    # fit model no training data
    # model = XGBClassifier(use_label_encoder=False,
    #                       n_estimators=75,
    #                       max_depth=5,
    #                       random_state=seed)
    # model.train(params, X_train, y_train, num_boost_round=num_boost_round)

    # make predictions for test data
    y_pred = model.predict(dtest)
    predictions = [round(value) for value in y_pred]
    #
    # # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
