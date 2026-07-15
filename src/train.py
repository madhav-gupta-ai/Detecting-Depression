
# Model training, Optuna hyperparameter tuning and evaluation
# For every feature CSV in features/word_embeddings and features/transformer:
# SMOTE-balances, tunes and evaluates Random Forest, Logistic Regression and
# a Multi-Layer Perceptron (100 Optuna trials each, 5-fold stratified CV on F1),
# appending to results/results.csv. format_results() then flattens the
# hyperparameter dictionaries into columns and saves results/final_results.csv.
#
# Run from the repository root:  python src/train.py

def get_results():

    import pandas as pd
    import os
    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import f1_score, accuracy_score
    from imblearn.over_sampling import SMOTE
    import optuna
    from tqdm import tqdm

    # Function to load CSV files from a given folder
    def load_csv_files(folder_path):
        files = []
        for filename in os.listdir(folder_path):
            if filename.endswith('.csv'):
                files.append(os.path.join(folder_path, filename))
        return files

    # Function to read a CSV file and split into training and testing sets
    def prepare_data(file_path):
        df = pd.read_csv(file_path)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    # Function to perform Optuna hyperparameter tuning for Random Forest
    def tune_random_forest(X_train, y_train):
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': None,
                'min_samples_split': 2, #trial.suggest_int('min_samples_split', 2, 5),
                'min_samples_leaf': 1, #trial.suggest_int('min_samples_leaf', 1, 4),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                'criterion': trial.suggest_categorical('criterion', ['entropy', 'gini']),
                'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']),
                'bootstrap': True
            }
            model = RandomForestClassifier(**params, random_state=42)
            skf = StratifiedKFold(n_splits=5)
            return cross_val_score(model, X_train, y_train, cv=skf, scoring='f1').mean()

        study = optuna.create_study(direction='maximize', study_name='Optimizing Random Forest')
        study.optimize(objective, n_trials=100)
        return study.best_params

    # Function to perform Optuna hyperparameter tuning for Logistic Regression
    def tune_logistic_regression(X_train, y_train):
        def objective(trial):
            params = {
                'C': trial.suggest_loguniform('C', 0.001, 100),
                'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']),
                'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
                'solver': trial.suggest_categorical('solver', ['liblinear', 'saga'])
            }
            model = LogisticRegression(max_iter=1000, **params, random_state=42)
            skf = StratifiedKFold(n_splits=5)
            return cross_val_score(model, X_train, y_train, cv=skf, scoring='f1').mean()

        study = optuna.create_study(direction='maximize', study_name='Optimizing Logistic Regression')
        study.optimize(objective, n_trials=100)
        return study.best_params

    # Function to perform Optuna hyperparameter tuning for Multi Layer Perceptron
    def tune_neural_network(X_train, y_train):
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
            skf = StratifiedKFold(n_splits=5)
            return cross_val_score(model, X_train, y_train, cv=skf, scoring='f1').mean()

        study = optuna.create_study(direction='maximize', study_name='Optimizing Neural Network')
        study.optimize(objective, n_trials=100)
        return study.best_params

    # Function to train the model and evaluate it on the test set
    def train_and_evaluate(model, X_train, y_train, X_test, y_test):
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        return {
            'F1-Score': f1_score(y_test, predictions),
            'Accuracy': accuracy_score(y_test, predictions)
        }

    # Function to process each CSV file in a folder and save results
    def process_files(folder_path, results_file):
        files = load_csv_files(folder_path)

        # Prepare a dictionary to store results
        all_results = {
            'Filename': [],
            'Model': [],
            'F1-Score': [],
            'Accuracy': [],
            'Hyperparameters': []
        }

        # Load existing results if the file exists
        if os.path.exists(results_file):
            results_df = pd.read_csv(results_file)
            all_results['Filename'] = results_df['Filename'].tolist()
            all_results['Model'] = results_df['Model'].tolist()
            all_results['F1-Score'] = results_df['F1-Score'].tolist()
            all_results['Accuracy'] = results_df['Accuracy'].tolist()
            all_results['Hyperparameters'] = results_df['Hyperparameters'].tolist()

        for file in files:
            if file not in all_results['Filename']:
                X_train, X_test, y_train, y_test = prepare_data(file)

                # Tuning and evaluating each model
                rf_params = tune_random_forest(X_train, y_train)
                lr_params = tune_logistic_regression(X_train, y_train)
                nn_params = tune_neural_network(X_train, y_train)

                rf_model = RandomForestClassifier(**rf_params, random_state=42)
                lr_model = LogisticRegression(max_iter=1000, **lr_params, random_state=42)
                nn_model = MLPClassifier(max_iter=1000, **nn_params, random_state=42)

                rf_performance = train_and_evaluate(rf_model, X_train, y_train, X_test, y_test)
                lr_performance = train_and_evaluate(lr_model, X_train, y_train, X_test, y_test)
                nn_performance = train_and_evaluate(nn_model, X_train, y_train, X_test, y_test)

                # Append results to the dictionary
                for model_name, performance, params in zip(
                    ['Random Forest', 'Logistic Regression', 'Neural Network'],
                    [rf_performance, lr_performance, nn_performance],
                    [rf_params, lr_params, nn_params]):

                    all_results['Filename'].append(file)
                    all_results['Model'].append(model_name)
                    all_results['F1-Score'].append(performance['F1-Score'])
                    all_results['Accuracy'].append(performance['Accuracy'])
                    all_results['Hyperparameters'].append(params)

                # Save the intermediate results to CSV
                pd.DataFrame(all_results).to_csv(results_file, index=False)
                print(f"Results updated for {file}")

    folders = ['features/word_embeddings', 'features/transformer']
    for folder in folders:
        process_files(folder, 'results/results.csv')



def format_results(results_file, modified_file):

    import pandas as pd
    import ast

    # Function to parse the dictionary of hyperparameters
    def parse_hyperparameters(hyperparams):
        return ast.literal_eval(hyperparams)

    df = pd.read_csv(results_file)
    parsed_hyperparams = df['Hyperparameters'].apply(parse_hyperparameters)

    hyperparams_df = pd.json_normalize(parsed_hyperparams)
    df = pd.concat([df.drop(columns=['Hyperparameters']), hyperparams_df], axis=1)

    # Save the modified DataFrame to a new CSV file
    # (fixed: previously wrote back to results_file)
    df.to_csv(modified_file, index=False)
    print(f"Results saved as {modified_file}")


if __name__ == "__main__":
    import os
    if not os.path.exists('results'):
        os.makedirs('results')
    get_results()
    format_results('results/results.csv', 'results/final_results.csv')
