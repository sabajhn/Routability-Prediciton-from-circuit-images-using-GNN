from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def do_sklearn_train(XX,YY, train_ids, test_ids):
    X = XX.numpy()
    Y = np.array(YY)
    
    X_train = X[train_ids]
    X_test = X[test_ids]
    Y_train = Y[train_ids]
    Y_test = Y[test_ids]

    rf_model = GradientBoostingClassifier(n_estimators=100, max_features="auto", random_state=44)
    rf_model.fit(X_train, Y_train)

    

    print("Accuracy score (training): {0:.3f}".format(rf_model.score(X_train, Y_train)))
    print("Accuracy score (validation): {0:.3f}".format(rf_model.score(X_test, Y_test)))
    y_hat = rf_model.predict(X_test)
    print(y_hat)

    TN = 0.0001
    FP = 0.0001
    FN = 0.0001
    TP = 0.0001
    for i in range(y_hat.shape[0]):
        if(y_hat[i] == 1 and Y_test[i] == 1):
            TP += 1
        if(y_hat[i] == 0 and Y_test[i] == 1):
            FN += 1
        if(y_hat[i] == 1 and Y_test[i] == 0):
            FP += 1
        if(y_hat[i] == 0 and Y_test[i] == 0):
            TN += 1

    accuracy = int( (TP + TN) / (TP + TN + FP + FN) * 1000) / 10.
    Prec = int( (TP) / (TP + FP) * 1000) / 10.
    Rec = int( (TP) / (TP + FN) * 1000) / 10.
    Spec = int( (TN) / (TN + FP) * 1000) / 10.
    print("performance: ", accuracy, Prec, Rec, Spec)
