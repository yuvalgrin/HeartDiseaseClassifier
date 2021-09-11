import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from CsvLoader import CsvLoader


def cross_validator(params, dtrain, num_boost_round):
    # This can take some time…
    min_mae = float("Inf")
    best_params = None
    for eta in [.3, .2, .1, .05, .01, .005]:
        print("CV with eta={}".format(eta))
        # Update parameters
        params['eta'] = eta
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
    print("Best params: {}, mean absolute error: {}".format(best_params, min_mae))


def main(data_set):
    # Split data into train and test sets
    seed = 7
    test_size = 0.1
    X_train, X_test, y_train, y_test = train_test_split(data_set.x_train, data_set.y_train,
                                                        test_size=test_size, random_state=seed)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    num_boost_round = 500
    params = {
        # Parameters that we are going to tune
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
    print("Best mean absolute error: {:.2f} in {} rounds".format(model.best_score, model.best_iteration + 1))

    # Use cross validation to find the best Hyper-parameters
    # cross_validator(X_train, y_train, X_test, y_test)

    # Make predictions for test data
    y_pred = model.predict(dtest)
    predictions = [round(value) for value in y_pred]

    # Evaluate the model predictions
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))