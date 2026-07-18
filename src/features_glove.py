
# GLoVe word-embedding feature extraction
# Reads data/Depression<suffix> and data/Control<suffix> and writes 8 feature CSVs
# (25/50/100/200 dimensions x sum/average).
# suffix = '_Clean' uses the keyword-cleaned dataset (as in the published study)
# and writes to features/word_embeddings/;
# suffix = '' uses the original posts and writes to features/word_embeddings_unclean/
# (used by the keyword-removal comparison study, src/cross_domain.py).
#
# Run from the repository root:  python src/features_glove.py
# (pass --original to embed the original, uncleaned posts instead)

def word_embedding(suffix):

    # Importing Libraries
    import os           # Used to import and traverse the system's directories
    import csv          # Used to read and write to a .csv file
    import re           # Used to implement Regular Expressions
    import numpy as np  # Working with arrays at a high level
    import gensim.downloader as api     # Loads the GloVe word embedding models
    from nltk.stem import WordNetLemmatizer     # Used during pre-processing
    from tqdm import tqdm                       # Loading bar functionality

    # Define the path to the folders containing the text files
    text_folder_path = 'data/Depression'+suffix
    control_folder_path = 'data/Control'+suffix

    # Output folder: the cleaned dataset is the published one and goes to
    # features/word_embeddings; the original posts go to a separate folder
    output_folder = 'features/word_embeddings' if suffix == '_Clean' else 'features/word_embeddings_unclean'

    # Get the list of text files in the folder
    text_files = [f for f in os.listdir(text_folder_path) if f.endswith(".txt")]
    control_files = [f for f in os.listdir(control_folder_path) if f.endswith(".txt")]

    # Function to find the word embedding of a single .txt file
    def findEmbedding(model, n_dimensions, text_file, folder_path, sum_or_average):

        # Open the text file and read its contents
        with open(os.path.join(folder_path, text_file), "r", encoding = 'utf-8') as f:
            text = f.read()

        # Data Pre-Processing:
        # Removing punctuations and numbers, splitting it into words, converting to lowercase,
        # removing stopwords and out-of-vocabulary words, and lemmatization using WordNet

        # Removing punctuations and numbers
        text = re.sub(r'[^a-zA-Z]+', ' ', text)

        # Converting to lowercase and removing stopwords and unknown words
        words = [word for word in text.lower().split() if word in model.key_to_index]

        # Lemmatizing using the WordNet Lemmatize
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words]

        # Sum up the word embeddings for each word in the text
        embedding_sum = np.array([0.0] * n_dimensions)

        # Finding the Word Embeddings for each word
        for i in range(len(lemmatized_words)):
            try:
                word_vector = model.get_vector(lemmatized_words[i])

                if sum_or_average == 'sum':
                    # Calculating the sum of word vectors
                    embedding_sum = [a + b for a, b in zip(embedding_sum, word_vector)]

                else:
                    # Calculating the average of word vectors
                    embedding_sum = [((i*a)/(i+1)) + (b/(i+1)) for a, b in zip(embedding_sum, word_vector)]

            except KeyError:
                pass

        return embedding_sum


    # Function to save the word embeddings of the .txt files from a folder
    def makeCSV(modelname, sum_or_average):

        # Creating the .csv file
        n_dimensions = int(modelname[14:])
        filename = f'{output_folder}/word_embeddings_{n_dimensions}_{sum_or_average}.csv'
        print(filename)

        # Loading the GloVe model
        model = api.load(modelname)

        with open(filename, "w", newline="", encoding = 'utf-8') as csv_file:
            writer = csv.writer(csv_file)

            for text_file in tqdm(text_files, desc="Depression Posts", total=len(text_files)):
                # Finding the embeddings of depression posts and saving
                embedding = findEmbedding(model, n_dimensions, text_file, text_folder_path, sum_or_average)
                writer.writerow(embedding + ['1'])  # class 1: depression

            for text_file in tqdm(control_files, desc="Control Posts", total=len(control_files)):
                # Finding the embeddings of control posts and saving
                embedding = findEmbedding(model, n_dimensions, text_file, control_folder_path, sum_or_average)
                writer.writerow(embedding + ['0'])  # class 0: control


    # Making seperate word embedding feature extractions for
    # 5 different GloVe Twitter models: 25, 50, 100 and 200 dimensions, and for each
    # on the basis of calculation by sum and calculation by average word vectors

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for modelname in [f'glove-twitter-{x}' for x in [25, 50, 100, 200]]:
        for sum_or_average in ['sum', 'average']:

            print(f"GloVe {modelname[14:]}-D Twitter Word Embeddings Model in use")
            makeCSV(modelname, sum_or_average)


if __name__ == "__main__":
    import sys
    word_embedding('' if '--original' in sys.argv else '_Clean')
