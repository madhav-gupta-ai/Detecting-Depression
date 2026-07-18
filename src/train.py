
# Model training, Optuna hyperparameter tuning and evaluation
# For every feature CSV in features/transformer and features/word_embeddings:
# SMOTE-balances the data and generates 5 independent 80-20 train/test splits,
# then tunes Random Forest, Logistic Regression and a Multi-Layer Perceptron
# with Optuna (100 trials each; objective = the mean of the 5-fold
# cross-validation F1 scores over the 5 training sets). Each tuned classifier
# is retrained on each of the 5 training sets and scored on the corresponding
# test set, and the mean and variance of the F1 scores and accuracies are
# recorded, along with the tuned hyperparameters, in results/final_results.csv.
# Feature files already present in the results file are skipped, so an
# interrupted run can be resumed.
#
# Warning: the full run is expensive (each Optuna trial performs 5-fold
# cross-validation on all 5 training sets) and can take days on an ordinary
# machine. To re-evaluate using the already-tuned hyperparameters in
# results/final_results.csv instead, use src/evaluate.py.
#
# Run from the repository root:  python src/train.py

def get_results():

    import csv
    import pandas as pd
    import numpy as np
    import os
    from sklearn.base import clone
    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import f1_score, accuracy_score
    from imblearn.over_sampling import SMOTE
    import optuna

    n_trials = 100      # Optuna trials per classifier
    n_splits = 5        # independent train/test splits per feature set

    # Columns of the results file (the hyperparameter dictionaries are flattened)
    columns = ['Filename', 'Model', 'F1-Score', 'Accuracy', 'F1-Variance', 'Accuracy-Variance',
               'n_estimators', 'max_features', 'criterion', 'class_weight', 'C', 'penalty',
               'solver', 'hidden_layer_sizes', 'activation', 'batch_size', 'learning_rate',
               'learning_rate_init']

    # Function to load CSV files from a given folder
    def load_csv_files(folder_path):
        files = []
        for filename in os.listdir(folder_path):
            if filename.endswith('.csv'):
                files.append(os.path.join(folder_path, filename))
        return files

    # Function to read a feature CSV, balance it with SMOTE, and generate the
    # 5 independent 80-20 train/test splits
    def prepare_data(file_path):
        df = pd.read_csv(file_path)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X, y)
        splits = []
        for k in range(1, n_splits + 1):
            splits.append(train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=k))
        return splits

    # Function to compute the tuning objective for a classifier: the mean of the
    # 5-fold cross-validation F1 scores over the training set of each split
    def mean_cv_f1(model, splits):
        scores = []
        for X_train, X_test, y_train, y_test in splits:
            skf = StratifiedKFold(n_splits=5)
            scores.append(cross_val_score(model, X_train, y_train, cv=skf, scoring='f1').mean())
        return sum(scores) / len(scores)

    # Function to perform Optuna hyperparameter tuning for Random Forest
    def tune_random_forest(splits):
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                'criterion': trial.suggest_categorical('criterion', ['entropy', 'gini']),
                'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']),
                'bootstrap': True
            }
            model = RandomForestClassifier(**params, random_state=42)
            return mean_cv_f1(model, splits)

        study = optuna.create_study(direction='maximize', study_name='Optimizing Random Forest')
        study.optimize(objective, n_trials=n_trials)
        return study.best_params

    # Function to perform Optuna hyperparameter tuning for Logistic Regression
    def tune_logistic_regression(splits):
        def objective(trial):
            params = {
                'C': trial.suggest_loguniform('C', 0.001, 100),
                'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']),
                'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
                'solver': trial.suggest_categorical('solver', ['liblinear', 'saga'])
            }
            model = LogisticRegression(max_iter=1000, **params, random_state=42)
            return mean_cv_f1(model, splits)

        study = optuna.create_study(direction='maximize', study_name='Optimizing Logistic Regression')
        study.optimize(objective, n_trials=n_trials)
        return study.best_params

    # Function to perform Optuna hyperparameter tuning for Multi Layer Perceptron
    def tune_neural_network(splits):
        def objective(trial):
            hidden_layer_sizes = trial.suggest_categorical(
                'hidden_layer_sizes', [
                    (256, 128, 128, 64, 64),
                    (256, 128, 64, 64, 32),
                    (256, 128, 64, 32, 32),
                    (256, 128, 64, 32, 16)
                ])
            params = {
                'hidden_layer_sizes': hidden_layer_sizes,
                'activation': trial.suggest_categorical('activation', ['relu', 'tanh', 'logistic']),
                'solver': 'adam',
                'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256, 512, 1024]),
                'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 'invscaling', 'adaptive']),
                'learning_rate_init': trial.suggest_loguniform('learning_rate_init', 0.001, 0.1)
            }
            model = MLPClassifier(max_iter=1000, **params, random_state=42)
            return mean_cv_f1(model, splits)

        study = optuna.create_study(direction='maximize', study_name='Optimizing Neural Network')
        study.optimize(objective, n_trials=n_trials)
        return study.best_params

    # Function to train a classifier on each of the 5 training sets and evaluate
    # it on the corresponding test sets
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

    # Function to save the accumulated result rows to the results file
    def save_results(results, results_file):
        with open(results_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=columns, restval='')
            writer.writeheader()
            for row in results:
                writer.writerow({key: ('' if value is None else value) for key, value in row.items()})

    # Function to process each CSV file in a folder and append the results
    def process_files(folder_path, results_file):
        files = load_csv_files(folder_path)

        # Load the existing results if the file exists (allows resuming)
        results = []
        if os.path.exists(results_file):
            with open(results_file, 'r', newline='', encoding='utf-8') as f:
                results = list(csv.DictReader(f))
        done_files = {row['Filename'] for row in results}

        for file in files:
            # Filenames are recorded relative to the features folder
            file_id = os.path.basename(folder_path) + '/' + os.path.basename(file)
            if file_id in done_files:
                continue

            print(f"\nProcessing {file_id}")
            splits = prepare_data(file)

            # Tuning each model
            rf_params = tune_random_forest(splits)
            lr_params = tune_logistic_regression(splits)
            nn_params = tune_neural_network(splits)

            rf_model = RandomForestClassifier(**rf_params, random_state=42)
            lr_model = LogisticRegression(max_iter=1000, **lr_params, random_state=42)
            nn_model = MLPClassifier(max_iter=1000, **nn_params, random_state=42)

            # Evaluating each tuned model over the 5 train/test splits
            for model_name, model, params in zip(
                ['Random Forest', 'Logistic Regression', 'Neural Network'],
                [rf_model, lr_model, nn_model],
                [rf_params, lr_params, nn_params]):

                performance = train_and_evaluate(model, splits)
                results.append({'Filename': file_id, 'Model': model_name, **performance, **params})

            # Save the intermediate results to CSV
            save_results(results, results_file)
            print(f"Results updated for {file_id}")

    folders = ['features/transformer', 'features/word_embeddings']
    for folder in folders:
        process_files(folder, 'results/final_results.csv')


if __name__ == "__main__":
    import os
    if not os.path.exists('results'):
        os.makedirs('results')
    get_results()
