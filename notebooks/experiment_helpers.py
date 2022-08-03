from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import roc_auc_score


import numpy as np
import pandas as pd

random_state = 123
pipeline_dummy = Pipeline(
    [('most_frequent', DummyClassifier(strategy='most_frequent'))]
)
pipeline_lr = Pipeline(
    [('scalar1',StandardScaler()), ('lr_classifier', LogisticRegression(
        random_state=random_state, solver='liblinear'))])
pipeline_dt = Pipeline(
    [('scalar2',StandardScaler()), ('dt_classifier', DecisionTreeClassifier(random_state=random_state))])
pipeline_lgb = Pipeline(
    [('scalar3',StandardScaler()), ('lgbm_classifier', lgb.LGBMClassifier(random_state=random_state))]
)

default_pipelines = [
    (pipeline_dummy, 'Most Frequent'),
    (pipeline_lr, 'LR'),
    (pipeline_dt, 'Decision Tree'),
    (pipeline_lgb, 'LGBM'),
]

def predict(
    X_trn, y_trn, X_test, y_test, 
    labels, pipelines
): 
    """ 
    X
    y_all - dataframe. Each column is a set of labels.
    X_test - optional test set. e.g. to split by date.
    y_test - optional test.
    test_size -  if X_test and y_test are none, the number of examples to randomly include in test set
    random_state - seed for splitting.
    labels - labels to consider.
    """
    rows = []
    for c, label in enumerate(labels):
        y_trn_c = y_trn.loc[:, label].astype('int')
        y_test_c = y_test.loc[:, label].astype('int')
        
        pipelines_c = None
        if np.mean(y_trn_c) == 0 or np.mean(y_trn_c) == 1:
            pipelines_c = [
                ('pipeline_constant', Pipeline(
                    [('constant', DummyClassifier(strategy='constant',constant=np.mean(y_trn_c)))]
                ))
            ]
        else:
            pipelines_c = pipelines

        for (pipe, _) in pipelines:
                pipe.fit(X_trn, y_trn_c)
        # print(f"== trn: {len(y_trn_c)},  test: {len(y_test_c)}")

        for i, (model, model_name) in enumerate(pipelines):
            preds = model.predict(X_test)
  
            row = {
                'label': label,
                'model_name': model_name,
                'trn_mean': y_trn_c.mean(),
                'test_mean': y_test_c.mean(),
                'num_trn': len(y_trn_c),
                'num_test': len(y_test_c),
            }
            row['accuracy'] = accuracy_score(y_test_c, preds)
            try:
                row['roc_auc'] = roc_auc_score(y_test_c, preds)
            except ValueError:
                row['roc_auc'] = None
            for score_name, score_func in [
                ('precision', precision_score),
                ('recall', recall_score),
                ('f1', f1_score)
            ]:
                try:
                    row[score_name] = score_func(y_test_c, preds, zero_division=0)
                except ValueError:
                    row[score_name] = None

            rows.append(row)
    return rows

def preprocess(
    df, nlp, 
    trn_mask, test_mask,
    labels, features, 
    words, tfidf_max_features=300,
    classify=True
):
    """
    Take in the notes or tweets data as provided by twitter and output
    a "ready for sklearn" version.

    Args - meant to match the output of the consolidate_files() function
        df - either a notes_with_ratings df or tweets_with_notes df
        text_col - name of the column with cleaned text
        df_type - 'notes_with_ratings' or 'tweets_with_notes'
        trn_mask - boolean array ("mask") that indicates which rows are used for training

    Output
        Dictionary with train and test sets
    """
    text_col = 'clean_txt'

    word_column_names = None
    if words == 'vecs':
        # # Run spaCy pre-processing and turn into numpy array
        docs = nlp.pipe(df[text_col])
        X = np.array([x.vector for x in docs])
    elif words == 'tfidf':
        vectorizer = TfidfVectorizer(max_features=tfidf_max_features)
        X = vectorizer.fit_transform(df[text_col].values).toarray()
        word_column_names = vectorizer.get_feature_names_out()
    else:
        X = None

    if X is not None:
        X = np.concatenate(
            [X, df[features].to_numpy()],
            axis=1
        )
    else:
        X = df[features].to_numpy()
    y_df = df[labels].copy()

    if classify:
        for label in labels:
            y_df.loc[:, labels] = y_df.loc[:, labels] >= 0.5

    if len(X) != len(y_df):
        print(len(X), "   ", len(y_df))
        raise Exception("length is not equal")

 
    X_trn = X[trn_mask, :]
    X_test = X[test_mask, :]

    y_trn = y_df.loc[trn_mask]
    y_test = y_df.loc[test_mask]
    return {
        'X_trn': X_trn,
        'X_test': X_test,
        'y_trn': y_trn,
        'y_test': y_test,
    }


def run_experiments(
    data, nlp, 
    trn_mask, test_mask,
    labels, features,
    pipelines,
    experiment_identifiers,
    words='vecs', 
    
):
    experiment_identifiers['words'] = words
    experiment_identifiers['features'] = ','.join(features)

    d = preprocess(
        data, nlp,
        labels=labels, features=features, words=words,
        trn_mask=trn_mask,
        test_mask=test_mask,
    )

    rows = predict(
        d['X_trn'], d['y_trn'],
        d['X_test'], d['y_test'],
        labels=labels,
        pipelines=pipelines
    )
    for k,v in experiment_identifiers.items():
        for row in rows:
            row[k] = v
    return rows