import os
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

# support both module and direct execution
if __name__ == '__main__' and __package__ is None:
    import sys
    import pathlib
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
    from src.data_prep import load_data, preprocess
else:
    from .data_prep import load_data, preprocess

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODEL_DIR, exist_ok=True)


def train_and_save(random_state=42):
    df = load_data()
    X, y, preprocessor, le, num_cols, cat_cols = preprocess(df, fit_label=True)

    # build pipeline
    clf = RandomForestClassifier(random_state=random_state, n_jobs=-1)
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', clf),
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)

    # quick grid for reasonable params
    param_grid = {
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [None, 10, 20],
    }

    gs = GridSearchCV(pipe, param_grid, cv=3, n_jobs=-1, verbose=1)
    gs.fit(X_train, y_train)

    print('Best params:', gs.best_params_)

    y_pred = gs.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print('Test accuracy:', acc)
    print('\nClassification report:')
    print(classification_report(y_test, y_pred))

    # save model pipeline and label encoder
    model_path = os.path.join(MODEL_DIR, 'obesity_pipeline.joblib')
    le_path = os.path.join(MODEL_DIR, 'label_encoder.joblib')
    joblib.dump(gs.best_estimator_, model_path)
    joblib.dump(le, le_path)
    print(f'Saved model to {model_path}')
    print(f'Saved label encoder to {le_path}')

    return gs.best_estimator_, le, acc


if __name__ == '__main__':
    train_and_save()
