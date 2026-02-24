import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'Obesity.csv')


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    return df


def build_preprocessor(X):
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

    # numeric scaler
    numeric_transformer = StandardScaler()

    # categorical encoder (support sklearn versions that use `sparse_output`)
    try:
        # sklearn >= 1.2
        categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        # older sklearn
        categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols),
        ],
        remainder='drop',
    )
    return preprocessor, numeric_cols, categorical_cols


def preprocess(df, fit_label=True):
    df = df.copy()

    # create BMI and drop Height/Weight
    df['BMI'] = df['Weight'] / (df['Height'] ** 2)
    df = df.drop(columns=['Height', 'Weight'])

    # target
    target = 'Obesity'
    y = df[target].astype(str)

    # features
    X = df.drop(columns=[target])

    # map binary yes/no to 1/0 where applicable
    for col in X.columns:
        if X[col].dtype == 'object':
            uniques = set([str(v).strip().lower() for v in X[col].dropna().unique()])
            if uniques <= {'yes', 'no'}:
                X[col] = X[col].str.strip().str.lower().map({'yes': 1, 'no': 0})

    # label encode target
    le = LabelEncoder()
    if fit_label:
        y_enc = le.fit_transform(y)
    else:
        y_enc = le.transform(y)

    preprocessor, numeric_cols, categorical_cols = build_preprocessor(X)

    return X, y_enc, preprocessor, le, numeric_cols, categorical_cols


if __name__ == '__main__':
    df = load_data()
    X, y, preprocessor, le, num_cols, cat_cols = preprocess(df)
    print('X shape:', X.shape)
    print('Numeric cols:', num_cols)
    print('Categorical cols:', cat_cols)
