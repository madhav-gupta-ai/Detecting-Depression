# Detecting Depression: Employing Word-Embeddings and Sentence Transformers

Code for the paper:

> Madhav Gupta, Mitali Balki, Sairaj Patki and Jayaraman K. Valadi, **"Detecting
> Depression: Employing Word-Embeddings and Sentence Transformers,"** *Journal
> of Computational Social Science*, vol. 9, no. 2, art. no. 38, May 2026.
> DOI: [10.1007/s42001-026-00467-2](https://doi.org/10.1007/s42001-026-00467-2)

Binary text classification of Reddit posts: given a post, predict whether its
author is a diagnosed Depression sufferer (291 posts) or a control user (698
posts). The dataset was collected with diagnosis-verification rules adapted
from the RSDD protocol and then cleaned: psychiatric keywords ("depression",
"diagnosis", "therapist" and their variants) are removed so classifiers cannot
rely on explicit clue words. Features come from GLoVe Twitter word-embeddings
(25/50/100/200 dimensions, summed or averaged per post) and four pre-trained
sentence transformers; Random Forest, Logistic Regression and Multi-Layer
Perceptron classifiers are tuned with Optuna after SMOTE class balancing. The
TF-IDF baseline is from the preceding conference paper, ["Detecting Depression:
Employing Natural Language Processing and Random
Forests"](https://doi.org/10.1007/978-981-97-8096-9_8) (CSCT 2023). The
repository holds the dataset, every pipeline stage, the recorded results, and
two trained models ready for inference.

## Setup

Python 3.12:

```
pip install -r requirements.txt
python -c "import nltk; [nltk.download(p) for p in ['wordnet', 'omw-1.4', 'punkt', 'punkt_tab', 'stopwords']]"
```

The trained models ship with the repository (`models/`), so inference works out
of the box. The feature-extraction and inference scripts download the
pre-trained GLoVe models (~1.5 GB) and sentence transformers (~1 GB) on first
use. The committed dataset and its collection method are documented in
[data/README.md](data/README.md).

## Usage

Run every script from the repository root. The committed dataset, features and
results let you start at any stage; to redo everything from scratch, in order:

```
python src/collect.py                 # collect the dataset from Reddit (needs API credentials)
python src/clean.py                   # remove the psychiatric keywords
python src/features_glove.py          # GLoVe features
python src/features_transformers.py   # sentence-transformer features
python src/features_tfidf.py          # TF-IDF baseline features
python src/train.py                   # Optuna tuning + 5-split evaluation
python src/evaluate.py                # re-evaluation from the tuned hyperparameters
python src/cross_domain.py            # keyword-removal comparison study
```

To classify a text file with the shipped models:

```
python src/predict.py examples/example_depression.txt
```

| Option | Meaning |
|---|---|
| `--model FILE` | `predict.py`: model bundle to use (default: `models/all_mpnet_base_v2_random_forest.joblib`) |
| `--original` | feature scripts: embed the original posts into `features/*_unclean/` instead of the cleaned ones |
| `--save-models` | `evaluate.py`: retrain and save the two headline models to `models/` |
| `--save-misclassified` | `evaluate.py`: extract the posts misclassified by the best model |
| `--results FILE`, `--outdir DIR` | `evaluate.py`: hyperparameter source and output folder (defaults: `results/final_results.csv`, `results`) |
| `--out FILE` | `cross_domain.py`: output CSV path (default: `results/cross_domain_rf_results.csv`) |

`src/collect.py` needs Reddit API credentials in
`data/support/API_credentials.txt` (copy `API_credentials_example.txt` and fill
in your own). Reddit content changes over time, so a fresh crawl reproduces the
collection method rather than the committed dataset; the committed features and
results all derive from the committed posts.

`src/train.py` runs the full Optuna search (100 trials per classifier, each
trial cross-validated over all 5 training sets) and can take days on an
ordinary machine. It saves after every feature set and skips finished ones, so
an interrupted run resumes where it stopped. `src/evaluate.py` produces the
same evaluation in minutes by reusing the tuned hyperparameters recorded in
`results/final_results.csv`.

## Notebooks

- [results.ipynb](notebooks/results.ipynb): the result figures of the study, drawn from `results/`
- [demo.ipynb](notebooks/demo.ipynb): pipeline walkthrough on the committed dataset, ending with inference on the example posts

Both notebooks are committed fully executed, so the figures are viewable
directly on GitHub.

## Disclaimer on responsible use

The models label a text post as Depression or Control from linguistic signal
alone, and were built to study NLP methods on a small research dataset. A
prediction is not a diagnosis; only a clinician can make one. Do not use this
project to screen, profile or make decisions about real people.

## License

The code is released under the [MIT License](LICENSE), © 2023 Madhav Gupta.
The Reddit-derived text posts remain the property of their respective authors.
