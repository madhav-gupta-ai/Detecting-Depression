READ ME

Detecting Depression: Employing Word-Embeddings and Sentence Transformers
Madhav Gupta, Mitali Balki, Sairaj Patki, Jayaraman K. Valadi

Data and Code Files
This repository contains the dataset, scripts and derived features associated with
collecting, analyzing and training machine learning classifiers on Reddit posts.

Repository layout:

data/
- Depression: .txt posts obtained from r/depression on Reddit (the Depression class).
- Depression_Clean: "cleaned" versions of the Depression posts. Cleaning involves
  removing psychiatry-based keywords (such as Depression, therapy, psychiatrist etc.)
- Control: .txt posts obtained from r/casualconversation on Reddit (the Control class).
- Control_Clean: "cleaned" versions of the Control posts.
- support: handcrafted .txt files related to the collection and cleaning of the posts.
  - API_credentials_example.txt: template for the Reddit API credentials. Copy it to
    API_credentials.txt (gitignored) and fill in your own client id, secret and user agent.
  - condition_words.txt: keywords referring to Depression
  - diagnosis_words.txt: keywords referring to diagnoses
  - doctor_words.txt: keywords referring to psychiatrists
  - mh_patterns.txt: keywords referring to mental disorders
  - mh_subreddits.txt: list of Reddit subreddits related to mental health
  - negative_diagnosis.txt: phrases indicating that the user is not diagnosed with Depression
  - positive_diagnosis.txt: phrases indicating that the user is diagnosed with Depression

src/ (run every script from the repository root, e.g. python src/train.py)
- collect.py: dataset collection using the Reddit API (PRAW). Writes data/Depression
  and data/Control.
- clean.py: keyword cleaning. Writes data/Depression_Clean and data/Control_Clean.
- features_glove.py: GLoVe word-embedding features (25/50/100/200-D, sum + average).
  Writes features/word_embeddings/.
- features_transformers.py: sentence-transformer features for four pre-trained models
  (all-MiniLM-L6-v2, all-mpnet-base-v2, gtr-t5-base, sentence-t5-base).
  Writes features/transformer/.
- features_tfidf.py: TF-IDF baseline features with SMOTE balancing and
  mutual-information top-K% selection. Writes features/tfidf/.
- train.py: SMOTE + Optuna hyperparameter tuning (100 trials, 5-fold stratified CV)
  and evaluation of Random Forest, Logistic Regression and Multi-Layer Perceptron
  on every feature CSV. Writes results/results.csv and results/final_results.csv.

features/
- word_embeddings: GLoVe-based features of the cleaned posts in .csv format
  (two files - summed + averaged - for each of 25-D, 50-D, 100-D, 200-D).
- transformer: sentence-transformer features of the cleaned posts in .csv format
  (one file per chosen sentence transformer).

results/
- final_results.csv: classification performance (F1, Accuracy) and tuned
  hyperparameters for every feature set x model combination.
- misclassified/: posts misclassified by the best models, used for the qualitative
  error analysis in the published paper.

experiments/
- blackhole.py: unpublished 2024 experiment - Black Hole metaheuristic feature
  selection wrapped around the classifiers.

notebooks/
- demo/demo.ipynb: self-contained walkthrough of the pipeline (preprocess -> TF-IDF /
  GLoVe -> Random Forest) on a CSV version of the dataset.
- test.ipynb: prototype of the Optuna tuning pipeline (May 2024).

Requirements: see requirements.txt (Python 3; praw, pandas, numpy, nltk, scikit-learn,
tqdm, gensim, sentence-transformers, optuna, imblearn).

Published work:
- Gupta, Patki & Valadi (2024), "Detecting Depression: Employing Natural Language
  Processing and Random Forests" (CSCT 2023).
- Gupta, Balki, Patki & Valadi, "Detecting Depression: Employing Word-Embeddings and
  Sentence Transformers", Springer Journal of Computational Social Science.
