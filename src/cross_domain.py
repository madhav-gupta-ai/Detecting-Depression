
# Controlled comparison study: the effect of keyword removal (cleaning)
# For each transformer feature set and each GLoVe (average) feature set,
# SMOTE-balances the cleaned and original versions of the features and trains
# simple Random Forest classifiers under 5-fold cross validation for the four
# combinations of cleaned/original training and testing data, recording the
# mean and variance of the F1 scores and accuracies in
# results/cross_domain_rf_results.csv.
#
# The cleaned features are read from features/transformer and
# features/word_embeddings; the original (unclean) features are read from
# features/transformer_unclean and features/word_embeddings_unclean, generated
# by running the two feature extraction scripts with the --original flag.
#
# Run from the repository root:  python src/cross_domain.py

import argparse
import os

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

# The four experiments of the controlled comparison
experiments = [
    ('train_clean_test_unclean', 'clean', 'unclean'),
    ('train_unclean_test_clean', 'unclean', 'clean'),
    ('train_clean_test_clean', 'clean', 'clean'),
    ('train_unclean_test_unclean', 'unclean', 'unclean'),
]


# Function to list the (clean, unclean) feature CSV pairs of the study:
# the four sentence transformers and the four GLoVe average feature sets
def feature_pairs():
    pairs = []
    for filename in sorted(os.listdir('features/transformer')):
        if filename.endswith('.csv'):
            pairs.append((f'features/transformer/{filename}',
                          f'features/transformer_unclean/{filename}'))
    for filename in sorted(os.listdir('features/word_embeddings')):
        if filename.endswith('.csv') and 'average' in filename:
            pairs.append((f'features/word_embeddings/{filename}',
                          f'features/word_embeddings_unclean/{filename}'))
    return pairs


def main():
    parser = argparse.ArgumentParser(description='Keyword-removal controlled comparison')
    parser.add_argument('--out', default='results/cross_domain_rf_results.csv',
                        help='output CSV path')
    args = parser.parse_args()

    rows = []
    for clean_path, unclean_path in feature_pairs():
        print(f"Processing {os.path.basename(clean_path)}")

        # The clean and unclean feature files list the posts in the same order,
        # so row i of one corresponds to row i of the other
        clean_df = pd.read_csv(clean_path)
        unclean_df = pd.read_csv(unclean_path)
        y = clean_df.iloc[:, -1]

        # SMOTE-balance both versions of the features (as in src/train.py)
        X_clean, y_balanced = SMOTE(random_state=42).fit_resample(clean_df.iloc[:, :-1].to_numpy(), y)
        X_unclean, _ = SMOTE(random_state=42).fit_resample(unclean_df.iloc[:, :-1].to_numpy(), y)
        X = {'clean': X_clean, 'unclean': X_unclean}
        y = y_balanced

        row = {'file': os.path.basename(clean_path)}
        skf = StratifiedKFold(n_splits=5)
        folds = list(skf.split(X['clean'], y))

        for name, train_version, test_version in experiments:
            f1_scores = []
            accuracies = []
            for train_index, test_index in folds:
                model = RandomForestClassifier(random_state=42)
                model.fit(X[train_version][train_index], y.iloc[train_index])
                predictions = model.predict(X[test_version][test_index])
                f1_scores.append(f1_score(y.iloc[test_index], predictions))
                accuracies.append(accuracy_score(y.iloc[test_index], predictions))

            row[f'{name}_f1_mean'] = np.mean(f1_scores)
            row[f'{name}_f1_var'] = np.var(f1_scores)
            row[f'{name}_acc_mean'] = np.mean(accuracies)
            row[f'{name}_acc_var'] = np.var(accuracies)

        rows.append(row)

    out_folder = os.path.dirname(args.out)
    if out_folder and not os.path.exists(out_folder):
        os.makedirs(out_folder)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"Results saved to {args.out}")


if __name__ == "__main__":
    main()
