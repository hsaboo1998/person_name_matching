import numpy as np
from sklearn.linear_model import LogisticRegression as LR
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC
from xgboost import XGBClassifier as XGB

rfc_param_space = {
    "n_estimators": np.arange(10, 350, 50, dtype=int),
    "max_depth": np.arange(2, 12, 2, dtype=int),
    "min_samples_split": np.arange(4, 22, 2, dtype=int),
    "min_samples_leaf": np.arange(4, 22, 2, dtype=int),
    "ccp_alpha": np.arange(0, 0.005, 0.0005),
}

xgb_param_space = {
    "n_estimators": np.arange(100, 350, 50, dtype=int),
    "learning_rate": np.linspace(0.004, 0.009, 10),
    "max_depth": np.arange(4, 9, 1, dtype=int),
    "colsample_bytree": np.linspace(0.2, 0.9, 10),
    "subsample": np.linspace(0.1, 1, 10),
    "gamma": np.logspace(-1, 1.8, 10),
    "reg_lambda": np.logspace(-1, 1.8, 10),
    "reg_alpha": np.logspace(-1, 1.8, 10),
}

model_training_param_space = {
    "rfc": [
        RFC(
            random_state=42,
            n_jobs=-1,
        ),
        rfc_param_space,
    ],
    "xgb": [
        XGB(
            random_state=42,
            n_jobs=-1,
            importance_type="total_gain",
        ),
        xgb_param_space,
    ],
}