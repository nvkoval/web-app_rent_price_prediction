import numpy as np
from sklearn.metrics import root_mean_squared_log_error
from sklearn.model_selection import KFold

def rmsle_score(y_true, y_pred):
    """Calculate RMSLE score"""
    y_true_exp = np.expm1(y_true)
    y_pred_exp = np.expm1(y_pred)
    return root_mean_squared_log_error(y_true_exp, y_pred_exp)

def run_cv(model, X, y, folds=3, random_state=7):
    """Run cross-validation and return mean RMSLE score"""
    y = np.log1p(y)
    kf = KFold(n_splits=folds, shuffle=True, random_state=random_state)
    train_scores, val_scores = [], []

    for k, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        train_score = rmsle_score(y_train, model.predict(X_train))
        val_score = rmsle_score(y_val, model.predict(X_val))

        print(f'[Fold {k}] train_rmsle: {train_score:.4f}, val_rmsle: {val_score:.4f}')
        train_scores.append(train_score)
        val_scores.append(val_score)

    print(f'RMSLE: {np.mean(val_scores):.4f} ± {np.std(val_scores):.4f}')
    return np.mean(val_scores)
