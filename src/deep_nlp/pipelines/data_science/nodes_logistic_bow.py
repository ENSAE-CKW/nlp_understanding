import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score



def train(X_train_encoded, y_train):

    print("Start Scale")

    scaler = StandardScaler()
    scaler.fit(X_train_encoded)
    print("End Scale")

    # Model definition
    model = LogisticRegression(solver='saga', penalty="l1"
                               , n_jobs=-1)

    grid_params = {
        "C": np.logspace(-3, 3, 7)
    }

    gs = GridSearchCV(model
                      , param_grid=grid_params
                      , scoring= "neg_log_loss"
                      , verbose= 1)

    # Traning model with scaled data
    print("Model training phase")
    X_train_encoded_scale = scaler.transform(X_train_encoded)
    del X_train_encoded

    gs.fit(X_train_encoded_scale, y_train)
    del X_train_encoded_scale

    # RETURN THE BEST MODEL ENCOUNTERED
    return [gs.best_estimator_, scaler]


def evaluate(X_valid_encoded, y_valid, model, scaler):

    print("Validation phase")
    X_valid_encoded_scale = scaler.transform(X_valid_encoded)
    del X_valid_encoded
    y_pred = model.predict(X_valid_encoded_scale)
    print("AUC : {:.6f}".format(roc_auc_score(y_valid, y_pred)))
    del X_valid_encoded_scale
    pass