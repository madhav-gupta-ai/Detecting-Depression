READ ME

Detecting Depression: Employing Word-Embeddings and Sentence Transformers 
Madhav Gupta, Mitali Balki, Sairaj Patki, Jayaraman K. Valadi

Data and Code Files
This repository contains scripts, functions and supplementary files associated with collecting, analyzing and training machine learning classifiers on Reddit posts.

Folders:

Depression: Stores .txt posts obtained from r/depression on Reddit. These files act as our Depression posts.
Depression_Clean: Stores "cleaned" versions of the .txt posts from the Depression folder. Cleaning involves removing psychiatry-based keywords (such as Depression, therapy, psychiatrist etc.)
Control: Stores .txt posts obtained from r/casualconversation on Reddit. These files act as our Control posts.
Control_Clean: Stores "cleaned" versions of the .txt posts from the Control folder. Cleaning involves removing psychiatry-based keywords (such as Depression, therapy, psychiatrist etc.)

support: Stores handcrafted .txt files related to the collection and cleaning of our Reddit posts.
- API_credentials.txt: The Reddit API details used for the collection of this dataset
- condition_words.txt: Keywords referring to Depression
- diagnosis_words.txt: Keywords referring to diagnoses
- doctor_words.txt: Keywords referring to psychiatrists
- mh_patterns.txt: Keywords referring to mental disorders
- mh_subreddits.txt: List of Reddit subreddits related to mental health
- negative_diagnosis.txt: Phrases indicating that the user is not diagnosed with Depression
- positive_diagnosis.txt: Phrases indicating that the user is diagnosed with Depression

transformer: Stores sentence-transformer based features of our .txt posts in .csv format
- contains four .csv files, one for each of our chosen sentence transformers

word_embeddings: Stores GLoVe based features of our .txt posts in .csv format
- contains eight .csv files, two (summed + averaged) for each of our GLoVe variants (25-D, 50-D, 100-D, 200-D)

requirements.txt: List of Python dependencies to be installed for the smooth functionality and operation of our code

mastercode.py: A comprehensive Python script consisting of the various functions that our code comprises of
- dataset_collection(): Uses the specified data collection methodology to obtain and save Reddit posts (both Depression and Control) and save them as .txt files in their respective folders
- remove_words(): Cleans the obtained .txt posts and saves them in their respective Clean folders
- word_embedding(suffix): Generates GLoVe word-embedding features for the chosen folder. 'suffix' helps specify the folder to be chosen ('' or '_Clean')
- transformer(suffix): Generates sentence transformer features for the chosen folder. 'suffix' helps specify the folder to be chosen ('' or '_Clean')
- get_results(): Implements SMOTE and tunes, trains and evaluates machine learning models (Random Forest, Logistic Regression, Multi-Layer Perceptron) for each word-embedding and sentence-transformer features .csv file, and saves their respective performances and best hyperparameters as 'results.csv'.
- format_results(results_file, modified_file): Parses the results.csv file and saves it in a more readable .csv format (clear hyperparameter columns) as modified_file.

Ensure that all dependencies are installed, and required data files are available in the support folder.