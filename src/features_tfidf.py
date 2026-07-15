
# TF-IDF feature extraction (baseline used in the study; core method of the
# preceding CSCT 2023 conference paper "Detecting Depression: Employing
# Natural Language Processing and Random Forests")
# Reads data/Depression_Clean and data/Control_Clean, writes the full and
# SMOTE-balanced TF-IDF matrices plus mutual-information top-K% subsets and
# wordclouds to features/tfidf/.
#
# Run from the repository root:  python src/features_tfidf.py

import os
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import mutual_info_classif
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from wordcloud import WordCloud

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Step 1: Lowercasing
    text = text.lower()

    # Step 2: Tokenization
    words = nltk.word_tokenize(text)

    # Step 3: Removing Stop Words
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Step 4: Removing Special Characters and Punctuation
    words = [word for word in words if word.isalnum()]

    return ' '.join(words)

def makeWordCloud(text, title, save_path):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(8, 4))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.savefig(save_path)
    plt.show()

def mutual_info_feature_selection(X, y, K_percentage=0.2):
    K = int(K_percentage * X.shape[1])
    mutual_info_scores = mutual_info_classif(X, y)
    top_features_indices = (-mutual_info_scores).argsort()[:K]
    top_features = X.iloc[:, top_features_indices]
    return top_features

def balance_data(X, y):
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X, y)
    return X_balanced, y_balanced

def main():
    depression_path = 'data/Depression_Clean'
    control_path = 'data/Control_Clean'
    output_dir = 'features/tfidf'
    min_occurences = 3

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    depression_data = []
    control_data = []

    # Load and preprocess data from Depression_Clean folder
    print("Preprocessing Depression data...")
    for filename in os.listdir(depression_path):
        if filename.endswith('.txt'):
            with open(os.path.join(depression_path, filename), 'r', encoding='utf-8') as f:
                text = f.read()
                processed_text = preprocess_text(text)
                depression_data.append(processed_text)

    # Load and preprocess data from Control_Clean folder
    print("Preprocessing Control data...")
    for filename in os.listdir(control_path):
        if filename.endswith('.txt'):
            with open(os.path.join(control_path, filename), 'r', encoding='utf-8') as f:
                text = f.read()
                processed_text = preprocess_text(text)
                control_data.append(processed_text)

    # Generate word clouds for the full text features and save as PNGs
    full_text_depression = ' '.join(depression_data)
    full_text_control = ' '.join(control_data)
    makeWordCloud(full_text_depression, 'Full Text Features (Depression)', f'{output_dir}/full_text_depression_wordcloud.png')
    makeWordCloud(full_text_control, 'Full Text Features (Control)', f'{output_dir}/full_text_control_wordcloud.png')

    # Combine data from both classes for TF-IDF representation
    combined_data = depression_data + control_data

    # Create the TF-IDF matrix
    vectorizer = TfidfVectorizer(min_df=min_occurences)
    tfidf_matrix = vectorizer.fit_transform(combined_data)

    # Get feature names (words)
    feature_names = vectorizer.get_feature_names_out()
    print("Number of features:", len(feature_names))

    # Convert the TF-IDF matrix to a DataFrame
    df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

    # Add class labels to the DataFrame (1 for depression, 0 for control)
    df['Label'] = [1] * len(depression_data) + [0] * len(control_data)

    # Save the regular full TF-IDF matrix to a CSV file
    df.to_csv(f'{output_dir}/tfidf_scores.csv', index=False)
    print(f"Regular full TF-IDF matrix saved to {output_dir}/tfidf_scores.csv")

    # Balance data using SMOTE
    print("Balancing data using SMOTE...")
    X_balanced, y_balanced = balance_data(df.drop(columns=['Label']), df['Label'])
    df_balanced = pd.DataFrame(X_balanced, columns=feature_names)
    df_balanced['Label'] = y_balanced

    # Save the balanced TF-IDF matrix to a CSV file
    df_balanced.to_csv(f'{output_dir}/tfidf_scores_balanced.csv', index=False)
    print(f"Balanced TF-IDF matrix saved to {output_dir}/tfidf_scores_balanced.csv")

    # Save the top K features from balanced data to multiple CSV files for different K percentages
    k_percentages = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
    for k_percentage in k_percentages:
        top_features_k = mutual_info_feature_selection(df_balanced.drop(columns=['Label']), df_balanced['Label'], K_percentage=k_percentage)
        top_features_k['Label'] = df_balanced['Label']
        top_features_k.to_csv(f'{output_dir}/tfidf_best_balanced_k_{k_percentage}.csv', index=False)
        print(f"Top K features with K={k_percentage} saved to {output_dir}/tfidf_best_balanced_k_{k_percentage}.csv")

if __name__ == "__main__":
    main()
