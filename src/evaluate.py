
# Re-evaluation of the tuned classifiers (without re-running the Optuna search)
# Reads the tuned hyperparameters recorded in results/final_results.csv,
# rebuilds every classifier, and re-evaluates it over the 5 independent 80-20
# train/test splits (mean and variance of the F1 score and accuracy), writing
# the scores to <outdir>/evaluation.csv. This reproduces the evaluation stage
# of src/train.py in minutes instead of days.
#
# Optional flags:
#   --save-models         also retrain and save the two headline models to models/
#                         (All MPNet base v2 + Random Forest, and
#                          200-D GLoVe average + Multi-Layer Perceptron)
#   --save-misclassified  also save the posts misclassified by the best model
#                         (All MPNet base v2 + Random Forest) across the 5
#                         splits to <outdir>/misclassified/
#   --results FILE        results file with the tuned hyperparameters
#                         (default: results/final_results.csv)
#   --outdir DIR          output folder (default: results)
#
# Run from the repository root:  python src/evaluate.py

import argparse
import ast
import os

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

n_splits = 5    # independent train/test splits, as in src/train.py

# The two headline models exported by --save-models: the best overall model and
# the best word-embedding model of the study
headline_models = [
    ('transformer/transformer_all_mpnet_base_v2.csv', 'Random Forest'),
    ('word_embeddings/word_embeddings_200_average.csv', 'Neural Network'),
]

# The model whose errors are extracted by --save-misclassified (the best model)
best_model = ('transformer/transformer_all_mpnet_base_v2.csv', 'Random Forest')


# Function to rebuild a classifier from one row of the results file
def build_classifier(row):
    if row['Model'] == 'Random Forest':
        return RandomForestClassifier(
            n_estimators=int(row['n_estimators']),
            max_features=row['max_features'],
            criterion=row['criterion'],
            class_weight=None if pd.isna(row['class_weight']) else row['class_weight'],
            random_state=42)

    if row['Model'] == 'Logistic Regression':
        return LogisticRegression(
            C=float(row['C']),
            penalty=row['penalty'],
            solver=row['solver'],
            class_weight=None if pd.isna(row['class_weight']) else row['class_weight'],
            max_iter=1000,
            random_state=42)

    if row['Model'] == 'Neural Network':
        hidden = row['hidden_layer_sizes']
        return MLPClassifier(
            hidden_layer_sizes=ast.literal_eval(hidden) if isinstance(hidden, str) else hidden,
            activation=row['activation'],
            solver='adam',
            batch_size=int(row['batch_size']),
            learning_rate=row['learning_rate'],
            learning_rate_init=float(row['learning_rate_init']),
            max_iter=1000,
            random_state=42)

    raise ValueError(f"Unknown model: {row['Model']}")


# Function to read a feature CSV, balance it with SMOTE, and generate the
# 5 independent 80-20 train/test splits (identical to src/train.py)
def prepare_data(file_path):
    df = pd.read_csv(file_path)
    n_original = len(df)    # rows beyond this index are synthetic SMOTE samples
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X, y)
    splits = []
    for k in range(1, n_splits + 1):
        splits.append(train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=k))
    return splits, n_original


# Function to train a classifier on each of the 5 training sets and evaluate it
# on the corresponding test sets (identical scoring to src/train.py)
def train_and_evaluate(model, splits):
    f1_scores = []
    accuracies = []
    for X_train, X_test, y_train, y_test in splits:
        split_model = clone(model)
        split_model.fit(X_train, y_train)
        predictions = split_model.predict(X_test)
        f1_scores.append(f1_score(y_test, predictions))
        accuracies.append(accuracy_score(y_test, predictions))
    return {
        'F1-Score': f'{np.mean(f1_scores) * 100:.2f}%',
        'Accuracy': f'{np.mean(accuracies) * 100:.2f}%',
        'F1-Variance': np.var(f1_scores),
        'Accuracy-Variance': np.var(accuracies)
    }


# Function to describe how a feature file's embeddings are obtained, so that a
# saved model bundle knows how to embed new text (see src/predict.py)
def embedding_config(filename):
    transformer_models = {
        'transformer_all_minilm_l6_v2.csv': 'sentence-transformers/all-MiniLM-L6-v2',
        'transformer_all_mpnet_base_v2.csv': 'sentence-transformers/all-mpnet-base-v2',
        'transformer_gtr_t5_base.csv': 'sentence-transformers/gtr-t5-base',
        'transformer_sentence_t5_base.csv': 'sentence-transformers/sentence-t5-base',
    }
    base = os.path.basename(filename)
    if base in transformer_models:
        return {'type': 'sentence_transformer', 'model': transformer_models[base]}
    # word_embeddings_<dimensions>_<sum_or_average>.csv
    parts = base.replace('.csv', '').split('_')
    return {'type': 'glove', 'model': f'glove-twitter-{parts[2]}', 'aggregation': parts[3]}


# Function to retrain a classifier on the first train/test split and save it as
# a model bundle in models/
def save_model(row, splits):
    X_train, X_test, y_train, y_test = splits[0]
    model = build_classifier(row)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    bundle = {
        'classifier': model,
        'model_name': row['Model'],
        'feature_file': row['Filename'],
        'feature_columns': list(X_train.columns),
        'embedding': embedding_config(row['Filename']),
        'split_random_state': 1,
        'test_f1': f1_score(y_test, predictions),
        'test_accuracy': accuracy_score(y_test, predictions),
    }

    if not os.path.exists('models'):
        os.makedirs('models')
    name = embedding_config(row['Filename'])['model'].split('/')[-1].replace('-', '_').lower()
    if bundle['embedding']['type'] == 'glove':
        name += '_' + bundle['embedding']['aggregation']
    filename = f"models/{name}_{row['Model'].replace(' ', '_').lower()}.joblib"
    joblib.dump(bundle, filename, compress=3)
    print(f"Saved {filename} (test F1: {bundle['test_f1'] * 100:.2f}%)")


# Function to collect the posts misclassified by a classifier across the 5
# splits and save them (original and cleaned forms) to <outdir>/misclassified/
def save_misclassified(row, splits, n_original, outdir):
    # Feature CSV rows are the Depression posts followed by the Control posts,
    # in the same file order used during feature extraction
    depression_files = [f for f in os.listdir('data/Depression_Clean') if f.endswith('.txt')]
    control_files = [f for f in os.listdir('data/Control_Clean') if f.endswith('.txt')]

    misclassified = {'Depression': set(), 'Control': set()}
    for X_train, X_test, y_train, y_test in splits:
        model = build_classifier(row)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        for position, index in enumerate(X_test.index):
            if index >= n_original or predictions[position] == y_test.iloc[position]:
                continue    # synthetic SMOTE sample or correctly classified
            if index < len(depression_files):
                misclassified['Depression'].add(depression_files[index])
            else:
                misclassified['Control'].add(control_files[index - len(depression_files)])

    # Save each misclassified post (original + cleaned) and a concatenated file
    for label in ['Depression', 'Control']:
        for form, suffix in [('Original', ''), ('Clean', '_Clean')]:
            folder = os.path.join(outdir, 'misclassified', label, form)
            if not os.path.exists(folder):
                os.makedirs(folder)

            concatenated_path = os.path.join(outdir, 'misclassified',
                                             f'misclassified_{label}_{form}_concatenated.txt')
            with open(concatenated_path, 'w', encoding='utf-8') as concatenated:
                for filename in sorted(misclassified[label], key=lambda f: int(f[5:-4])):
                    with open(f'data/{label}{suffix}/{filename}', encoding='utf-8') as f:
                        content = f.read()
                    with open(os.path.join(folder, filename), 'w', encoding='utf-8') as f:
                        f.write(content)
                    concatenated.write(f'{filename}:\n{content}\n\n')
                    concatenated.write('-' * 38 + '\n')

    print(f"Saved {len(misclassified['Depression'])} misclassified Depression posts "
          f"and {len(misclassified['Control'])} misclassified Control posts "
          f"to {os.path.join(outdir, 'misclassified')}")


def main():
    parser = argparse.ArgumentParser(description='Re-evaluate the tuned classifiers')
    parser.add_argument('--results', default='results/final_results.csv',
                        help='results file with the tuned hyperparameters')
    parser.add_argument('--outdir', default='results', help='output folder')
    parser.add_argument('--save-models', action='store_true',
                        help='retrain and save the headline models to models/')
    parser.add_argument('--save-misclassified', action='store_true',
                        help='save the posts misclassified by the best model')
    args = parser.parse_args()

    results = pd.read_csv(args.results)
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    evaluation = []
    for filename, file_rows in results.groupby('Filename', sort=False):
        print(f"\nEvaluating {filename}")
        splits, n_original = prepare_data(os.path.join('features', filename))

        for _, row in file_rows.iterrows():
            performance = train_and_evaluate(build_classifier(row), splits)
            evaluation.append({'Filename': filename, 'Model': row['Model'], **performance})
            print(f"  {row['Model']}: F1 {performance['F1-Score']}, "
                  f"Accuracy {performance['Accuracy']}")

            if args.save_models and (filename, row['Model']) in headline_models:
                save_model(row, splits)
            if args.save_misclassified and (filename, row['Model']) == best_model:
                save_misclassified(row, splits, n_original, args.outdir)

    evaluation_path = os.path.join(args.outdir, 'evaluation.csv')
    pd.DataFrame(evaluation).to_csv(evaluation_path, index=False)
    print(f"\nEvaluation saved to {evaluation_path}")


if __name__ == "__main__":
    main()
