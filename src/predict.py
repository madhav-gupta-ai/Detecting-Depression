
# Inference: predict whether a text post was written by a Depression sufferer
# Loads a trained model bundle from models/ and classifies the text in a given
# .txt file. The text goes through the same steps as the training data:
# psychiatric keywords are removed (as in src/clean.py), the text is embedded
# with the same model the classifier was trained with (a sentence transformer
# or GLoVe word-embeddings), and the classifier predicts the class label
# (1 = Depression, 0 = Control).
#
#   python src/predict.py path/to/post.txt
#   python src/predict.py path/to/post.txt --model models/glove_twitter_200_average_neural_network.joblib
#
# Note: this is a research demonstration, not a diagnostic tool.
#
# Run from the repository root.

import argparse
import os
import re
import sys

import joblib
import pandas as pd


# Function to remove the psychiatric keywords from a text (as in src/clean.py)
def clean_text(text):
    words = set()
    for words_file in ['condition_words.txt', 'doctor_words.txt', 'diagnosis_words.txt']:
        with open(f'data/support/{words_file}', 'r', encoding='utf-8') as f:
            words.update(line.strip() for line in f)
    return re.sub(r'\b(?:{})\b'.format('|'.join(map(re.escape, words))), '', text, flags=re.IGNORECASE)


# Function to embed a text with a sentence transformer
# (same chunking as src/features_transformers.py: texts of 256 words or more
# are split into segments of up to 200 words whose embeddings are averaged)
def embed_sentence_transformer(text, model_name):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)

    words = text.split()
    if len(words) < 256:
        return model.encode(text).tolist()

    n = int(len(words) / 200) + 1
    embeddings = None
    for i in range(n):
        segment = ' '.join(words[i * 200:(i + 1) * 200])
        segment_embedding = model.encode(segment).tolist()
        if embeddings is None:
            embeddings = segment_embedding
        else:
            embeddings = [a + b for a, b in zip(embeddings, segment_embedding)]
    return [value / n for value in embeddings]


# Function to embed a text with GLoVe word-embeddings
# (same pre-processing and aggregation as src/features_glove.py)
def embed_glove(text, model_name, sum_or_average):
    import gensim.downloader as api
    from nltk.stem import WordNetLemmatizer
    model = api.load(model_name)
    n_dimensions = model.vector_size

    # Removing punctuations and numbers, lowercasing, removing out-of-vocabulary
    # words, and lemmatization using WordNet
    text = re.sub(r'[^a-zA-Z]+', ' ', text)
    words = [word for word in text.lower().split() if word in model.key_to_index]
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]

    embedding_sum = [0.0] * n_dimensions
    for i in range(len(lemmatized_words)):
        try:
            word_vector = model.get_vector(lemmatized_words[i])
            if sum_or_average == 'sum':
                embedding_sum = [a + b for a, b in zip(embedding_sum, word_vector)]
            else:
                embedding_sum = [((i * a) / (i + 1)) + (b / (i + 1)) for a, b in zip(embedding_sum, word_vector)]
        except KeyError:
            pass
    return embedding_sum


def main():
    parser = argparse.ArgumentParser(description='Classify a text post (Depression vs Control)')
    parser.add_argument('textfile', help='path to a .txt file containing the post')
    parser.add_argument('--model', default='models/all_mpnet_base_v2_random_forest.joblib',
                        help='trained model bundle to use')
    args = parser.parse_args()

    if not os.path.exists(args.textfile):
        sys.exit(f"Error: file not found: {args.textfile}")
    with open(args.textfile, 'r', encoding='utf-8') as f:
        text = f.read()
    if not text.strip():
        sys.exit(f"Error: {args.textfile} is empty")

    if not os.path.exists(args.model):
        sys.exit(f"Error: model bundle not found: {args.model}\n"
                 "Generate the bundles with:  python src/evaluate.py --save-models")
    bundle = joblib.load(args.model)

    # The same cleaning and embedding steps used for the training data
    cleaned = clean_text(text)
    embedding = bundle['embedding']
    print(f"Embedding the text with {embedding['model']}...")
    if embedding['type'] == 'sentence_transformer':
        features = embed_sentence_transformer(cleaned, embedding['model'])
    else:
        features = embed_glove(cleaned, embedding['model'], embedding['aggregation'])

    X = pd.DataFrame([features], columns=bundle['feature_columns'])
    prediction = bundle['classifier'].predict(X)[0]
    probability = bundle['classifier'].predict_proba(X)[0][int(prediction)]

    label = 'Depression (1)' if int(prediction) == 1 else 'Control (0)'
    print(f"\nPrediction: {label}")
    print(f"Confidence: {probability * 100:.1f}%")
    print(f"Model: {bundle['model_name']} on {embedding['model']} features "
          f"(test F1: {bundle['test_f1'] * 100:.2f}%)")
    print("\nNote: this is a research demonstration, not a diagnostic tool.")


if __name__ == "__main__":
    main()
