from __future__ import annotations

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.dummy import DummyClassifier


def make_preprocessor(numeric_cols, categorical_cols):
    return ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical_cols),
    ], remainder='drop')


def make_model_zoo(random_state=42):
    models = {
        'dummy_prior': DummyClassifier(strategy='prior'),
        'logreg_balanced': LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0, random_state=random_state),
        'random_forest_balanced': RandomForestClassifier(n_estimators=500, class_weight='balanced_subsample', n_jobs=-1, random_state=random_state, min_samples_leaf=2),
        'extra_trees_balanced': ExtraTreesClassifier(n_estimators=700, class_weight='balanced', n_jobs=-1, random_state=random_state, min_samples_leaf=2),
    }
    try:
        from xgboost import XGBClassifier
        models['xgboost'] = XGBClassifier(
            n_estimators=500, max_depth=3, learning_rate=0.03, subsample=0.9, colsample_bytree=0.8,
            eval_metric='logloss', random_state=random_state, n_jobs=-1
        )
    except Exception:
        pass
    try:
        from lightgbm import LGBMClassifier
        models['lightgbm'] = LGBMClassifier(
            n_estimators=500, learning_rate=0.03, num_leaves=31, class_weight='balanced', random_state=random_state, n_jobs=-1
        )
    except Exception:
        pass
    return models


def build_pipeline(model, numeric_cols, categorical_cols):
    return Pipeline([
        ('preprocess', make_preprocessor(numeric_cols, categorical_cols)),
        ('model', model)
    ])
