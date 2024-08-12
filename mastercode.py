
# Master File

def dataset_collection():
    
    # Importing libraries
    import os       # For creating the folders
    import praw     # the official Reddit API used
    import hashlib  # provides the SHA-1 hash encoding used for user anonymity
    from prawcore.exceptions import Forbidden   # PRAW exception, to be handled
    
    from datetime import datetime, timedelta    # used to check date and timestamps
    from statistics import mean, median         # functions to compute averages
    from tqdm import tqdm                       # Loading bar
           

    # Some dataset specifications    
    n_users = 1000          # no. of users to consider at a time (PRAW max: 1000)
    n_posts_per_user = 5    # max. no. of posts to take from a single author
    n_recent_posts = None   # how far back in user history (None = the max 1000 posts)
    n_days_post_diag = 365  # how many days of posts to take after a diagnosis
    n_chars_per_post = 60   # min. no. of chars per post, 6 chars ~ 1 word + 1 punctuation
    

    # Extracting some lists of phrases from .txt files
    print("Initializing...\n")
    
    # List of Mental Health Subreddits of interest
    with open('support/mh_subreddits.txt') as file:
        # Read each line and store it in a list
        mh_subreddits = [line.strip().lower() for line in file]
    
    # List of search queries related to Diagnosis Posts   
    with open('support/diagnosis_words.txt') as file:
        search_queries = [line.strip().lower() for line in file]
    
    # List of Diagnosis Post phrases
    with open('support/positive_diagnosis.txt') as file:
        positive_diagnosis = [line.strip().lower() for line in file]
    
    # List of false-positive Diagnosis Post phrases
    with open('support/negative_diagnosis.txt') as file:
        negative_diagnosis = [line.strip().lower() for line in file]
    
    # List of phrases referring to Clinical Depression
    with open('support/condition_words.txt') as file:
        condition_words = [line.strip().lower() for line in file]
    
    # List of phrases related to Mental Health
    with open('support/mh_patterns.txt') as file:
        mh_patterns = [line.strip().lower() for line in file]
    

    # Create the Depression folder if it doesn't exist
    if not os.path.exists('Depression'):
        os.makedirs('Depression')

    # Create the Control folder if it doesn't exist
    if not os.path.exists('Control'):
        os.makedirs('Control')

    
    # Initializing dataset statistics
    depression_posts_count = len(os.listdir('Depression'))
    control_posts_count = len(os.listdir('Control'))
    user_count = 1
    total_depression_posts_count = []
    depression_posts_wordcounts = []
    
    
    # Loading the Reddit API Credentials
    with open('support/API_credentials.txt') as file:
        credentials = [line.strip() for line in file]
    
    reddit = praw.Reddit(client_id = credentials[0], 
                         client_secret = credentials[1], 
                         user_agent = credentials[2])
    
    
    # Function to compute the SHA-1 hash of an input text
    def findHash(text):
        
        # Create a SHA-1 hash of the text
        hash_object = hashlib.sha1(text.encode('utf-8'))
        
        # Get the hexadecimal representation of the hash
        hex_dig = hash_object.hexdigest()
        return hex_dig
    

    # Set of users to ignore (users already seen)
    # Also, adding the moderators (u/circinia and u/SQLwitch)
    userset = {findHash('circinia'), findHash('SQLwitch')}  
    
    
    # Function to check if a post is a Diagnosis Post
    # Diagnosis Post - a post in which the user claims to be diagnosed with depression
    def detectDiagnosis(post):
        
        # Reading the title and body text of the post        
        title, timestamp, subreddit, post_body = post
        post_text = title.lower() + ' ' + post_body.lower()
        
        # Strings in negative_diagnosis may cause false positives
        # such as "my mother was diagnosed...", "advice for diagnosis"
        # Thus, we immediately reject any posts with these strings        
        for line in negative_diagnosis:
            if line in post_text:
                return False
    
        # Strings in positive_diagnosis indicate a clinical diagnosis    
        for line in positive_diagnosis:
            if line in post_text:
                    
                    # Positive diagnosis string pattern exists in post
                    # Locating the first occurrence of such a string
                    for i in range(len(post_text)):
                        if post_text[i:i+len(line)] == line:
                            
                            # There must be a mention of depression 
                            # within 80 chatacters of this string
                            
                            context_words = ''
                            
                            for j in range(max(0, i-80), min(len(post_text), i+80)):
                                if post_text[j].isalpha():  # is an alphabetic character
                                    context_words += post_text[j]
                                else:
                                    context_words += ' '
                                    
                            for line2 in condition_words:
                                if line2 in context_words:
                                    
                                    # Diagnosis post condition satisfied
                                    print("Found Diagnosis Post")
                                    return True
        
        # Diagnosis post condition not satisfied
        return False
    
    # Function to check if a post mentions mental health at all
    def detectMH(post):
        
        # Reading the title and body text of the post
        title, timestamp, subreddit, post_body = post
        post_text = title.lower() + ' ' + post_body.lower()
        
        # Check if the post was posted to a mental health subreddit
        if subreddit in mh_subreddits:
            return True
        
        # Strings in mh_patterns are common phrases for mental health
        # Thus we reject control posts whose users have ever used these strings     
        for line in mh_patterns:
            if line in post_text:
                return True
        
        # Post does not mention MH
        return False
    
    # Function to collect all the text posts of an author: [(title, timestamp, subreddit, body text)]
    def getAllUserTextPosts(author):
        
        # Get all the posts made by the author
        all_posts = author.submissions.new(limit = n_recent_posts)
        
        # Filter the text posts and extract the relevant information
        text_posts = [(post.title, post.created_utc, post.subreddit.display_name, post.selftext) for post in all_posts if post.selftext != ""]
        
        # Sort the text posts by time from oldest to newest
        text_posts.sort(key = lambda x: x[1])
        
        return text_posts
    
    # Function to get all existing posts in the folder to ensure there are no duplicates
    def get_existing_posts(folder_path):
        existing_posts = []
        for filename in os.listdir(folder_path):
            if filename.endswith('.txt'):
                with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                    existing_posts.append(file.read())
        return existing_posts
    
    existing_depression_posts = get_existing_posts('Depression')
    existing_control_posts = get_existing_posts('Control')

    depression_posts_wordcounts = [len(post) for post in existing_depression_posts]
    

    # Searching and obtaining the search results for some search queries related 
    # to Diagnosis Posts from r/depression - to find possibly diagnosed users
    subreddit = reddit.subreddit('depression')
    print("Obtaining posts from r/depression")
    for search_query in search_queries:

        print(f"Searching {search_query}")
        for hot_post in tqdm(subreddit.search(search_query, sort='relevance', limit = n_users), desc=f"Searching users with {search_query}", total=n_users):
            
            # Making sure that the author's account exists (has not been deleted)
            if hot_post.author is None:
                continue    # look at the next post and its author
            
            # Anonymizing and saving the author as seen
            author_name = findHash(hot_post.author.name)
            author = reddit.redditor(hot_post.author.name)
            print(f"\nFrom user no. {user_count}: u/{author_name}")
            user_count+=1
            
            if author_name in userset:
                # Author was already seen before
                print("Author already looked at")
                continue
            
            else:
                # Recording the author as seen
                userset.add(author_name)
            
            # Going through the author's post history
            try:
                # Collecting all the posts of the user
                user_posts = getAllUserTextPosts(author)
                print(f"found {len(user_posts)} posts")
                
            except Forbidden:
                # In case of Forbidden access errors, we skip the user
                print("Error encountered")
                continue
            
            # Finding the first Diagnosis Post
            i = 0
            while i < len(user_posts):
                if detectDiagnosis(user_posts[i]):
                    print("Diagnosis Post Found")
                    break
                i+=1
                    
            if i == len(user_posts):
                # Diagnosis post not found, skipping this user
                print("No good posts found")
                continue
            
            # Diagnosis Post found
            # Removing all the posts before the diagnosis post
            user_posts = user_posts[i:]
            
            # Setting the start and end times for acceptable posts
            # From Diagnosis Posts till (n_days_post_diag) days after
            start_time = datetime.fromtimestamp(user_posts[0][1])
            end_time = start_time + timedelta(days = n_days_post_diag)
            
            # Looking at the posts made after the diagnosis
            counter = 0
            while counter < len(user_posts):
                post = user_posts[counter]
                
                if datetime.fromtimestamp(post[1]) > end_time:
                    # Reached the end of the time window
                    user_posts = user_posts[:counter]
                    break
                
                if (post[2] in mh_subreddits) and (len(post[3]) >= n_chars_per_post):
                    # Acceptable post for the dataset
                    counter+=1
                
                else:
                    # Post is either not MH-related or not verbose enough
                    user_posts = user_posts[:counter] + user_posts[(counter+1):]
                 
            # counter + 1 = No. of acceptable posts gotten from this user
            total_depression_posts_count += [counter+1]
            
            # Picking the (n_posts_per_user) most verbose posts    
            if ((counter+1) > n_posts_per_user):
                user_posts = (sorted(user_posts, key = lambda x: len(x[3]), reverse=True))[:n_posts_per_user]
                
    
            # Saving the post title and body text as .txt files    
            print(f"Found {len(user_posts)} acceptable posts\n")
            for i in range(len(user_posts)):             

                post_content = user_posts[i][0] + '\n' + user_posts[i][3]
                if post_content not in existing_depression_posts:   # not a duplicate

                    depression_posts_count+=1
                    depression_posts_wordcounts += [len(post_content)]
    
                    filename = f"Depression/post_{depression_posts_count}.txt"                
                    with open(filename, 'w', encoding='utf-8') as file:
                        file.write(post_content)

                    # Add the new post content to the existing posts list
                    existing_depression_posts.append(post_content)
                
                else:
                    print("Duplicate found")
                    

    # Get hot (trending) posts from r/CasualConversation (control group)    
    subreddit = reddit.subreddit('casualconversation')
    print("Obtaining control posts from r/casualconversation")
    for hot_post in tqdm(subreddit.hot(limit = n_users), desc="Searching for Control users", total=n_users):
        
        # Making sure that the author's account exists (has not been deleted)
        if hot_post.author is None:
            continue
        
        # Looking at the post history of the author
        author_name = findHash(hot_post.author.name)
        author = reddit.redditor(hot_post.author.name)
        print(f"\nFrom user no. {user_count}: u/{author_name}")
        user_count+=1
        
        if author_name in userset:
            # Author's posts already obtained
            print("Author already looked at")
            continue
        
        elif len(hot_post.selftext) < n_chars_per_post:
            print("Post too short")
            continue
        
        else:
            # Recording the author as seen
            userset.add(author_name)
        
        try:
            # Collecting all the posts of the user
            user_posts = getAllUserTextPosts(author)
            print(f"found {len(user_posts)} posts")
            
        except Forbidden:
            # In case of Forbidden access errors, we skip the user
            print("Error encountered")
            continue
        
        # Checking for posts containing mentions of mental health
        i = 0
        while i < len(user_posts):
            
            if detectMH(user_posts[i]):
                print("MH Mention Post Found")
                break
            i+=1
                
        if i != len(user_posts):
            # User may have a history of mental health, reject control post
            continue

        post_content = hot_post.title + '\n' + hot_post.selftext
        if post_content in existing_control_posts:
            print("Duplicate found")
            continue
        
        # Saving the post title and body text as .txt files
        print("Saving the post")
        control_posts_count+=1
        filename = f"Control/post_{control_posts_count}.txt"
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(post_content)
        existing_control_posts.append(post_content)    
    
            
    # Printing the depression posts findings and statistics        
    print("\nTotal no. of posts found:", sum(total_depression_posts_count))
    print("Mean posts per user:", mean(total_depression_posts_count))
    print("Median number of posts:", median(total_depression_posts_count))        
    print("Mean length of posts:", mean(depression_posts_wordcounts), "chars")
    print("Median length of posts:", median(depression_posts_wordcounts), "chars")
        
# dataset_collection()



def remove_words():
    
    import os
    import re
    
    def remove_words_in_folder(input_folder, output_folder, words_files):
        # Create the output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
    
        # Read the words to remove from the words files
        words = set()
        for words_file in words_files:
            with open(f'support/{words_file}', 'r', encoding = 'utf-8') as f:
                words.update(line.strip() for line in f)
    
        # Iterate over the files in the input folder
        for filename in os.listdir(input_folder):
            input_file = os.path.join(input_folder, filename)
            output_file = os.path.join(output_folder, filename)
    
            # Skip subdirectories and non-txt files
            if os.path.isdir(input_file) or not filename.endswith('.txt'):
                continue
    
            # Open the input and output files
            with open(input_file, 'r', encoding = 'utf-8') as input_f, open(output_file, 'w', encoding = 'utf-8') as output_f:
                # Read each line in the input file
                for line in input_f:
                    # Remove words from the line using regular expressions
                    cleaned_line = re.sub(r'\b(?:{})\b'.format('|'.join(map(re.escape, words))), '', line, flags=re.IGNORECASE)
    
                    # Write the cleaned line to the output file
                    output_f.write(cleaned_line)
    
            print(f"Processed: {input_file}")
    
    # Specify the input folder containing the .txt files
    input_folder_depression = 'Depression'
    input_folder_control = 'Control'
    
    # Specify the output folder where the copies will be created
    output_folder_depression = 'Depression_Clean'
    output_folder_control = 'Control_Clean'
    
    # Specify the files containing the words to remove
    words_files = ['condition_words.txt', 'doctor_words.txt', 'diagnosis_words.txt']
    
    # Call the function to remove words in the depression folder
    remove_words_in_folder(input_folder_depression, output_folder_depression, words_files)
    
    # Call the function to remove words in the control folder
    remove_words_in_folder(input_folder_control, output_folder_control, words_files)

# remove_words()



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
    text_folder_path = 'Depression'+suffix
    control_folder_path = 'Control'+suffix
    
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
        filename = f'word_embeddings/word_embeddings_{n_dimensions}_{sum_or_average}.csv'
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
                writer.writerow(embedding + ['0'])  # class 2: control
            
    
    # Making seperate word embedding feature extractions for
    # 5 different GloVe Twitter models: 25, 50, 100 and 200 dimensions, and for each 
    # on the basis of calculation by sum and calculation by average word vectors
    
    # Create the output folder if it doesn't exist
    if not os.path.exists('word_embeddings'):
        os.makedirs('word_embeddings')

    for modelname in [f'glove-twitter-{x}' for x in [25, 50, 100, 200]]:
        for sum_or_average in ['sum', 'average']:
            
            print(f"GloVe {modelname[14:]}-D Twitter Word Embeddings Model in use")        
            makeCSV(modelname, sum_or_average)
        
# word_embedding('_Clean')



def transformer(suffix):

    # Code for converting cleaned text files to vectors
    import os
    from sentence_transformers import SentenceTransformer
    import pandas as pd
    from tqdm import tqdm

    # function to create sentence embeddings
    def create_embeddings(text, model):

        # config
        separator = ' '

        # check if text is above 256 words
        if len(text.split()) < 256:

            # create the embedding
            embeddings = model.encode(text).tolist()

        else:

            #print("entered loop")

            # find the number of parts to split text into
            n = int(len(text.split())/200) + 1

            # run this loop to create embeddings for the different parts
            for i in range(n):
                #print(i)

                # if it is the first run of the loop
                if i == 0:

                    # split the sentence into previous and leftover
                    text_p = text.split()[:200]
                    text_l = text.split()[200:]

                    # convert text_p back to text to embedd it
                    text_p = separator.join(text_p)

                    # embedd text_p
                    embeddings = model.encode(text_p).tolist()
                    #print(f"first loop: {embeddings[0]}")

                # for round 1 onwards, once embeddings have been initialized

                # if it is the last round do the following
                if i == (n-1):

                    # convert text_l back to text to embedd it
                    text_l = separator.join(text_l)

                    # embedd leftover text_l
                    embedding_p = model.encode(text_l).tolist()

                    # add it to the previous embedding to get the embeddings
                    temp_embed = [(embeddings[j]+embedding_p[j]) for j in range(len(embedding_p))]
                    # update embeddings
                    embeddings = temp_embed
                    #print(f"last loop: {embeddings[0]}")
                    # if not last round

                elif i > 0:

                    # split the sentence into previous and leftover
                    text_p = text_l[:200]
                    text_l = text_l[200:]

                    # convert text_p back to text to embedd it
                    text_p = separator.join(text_p)

                    # embedd text_p
                    embedding_p = model.encode(text_p).tolist()

                    # add it to the previous embedding to get the embeddings
                    temp_embed = [(embeddings[j]+embedding_p[j]) for j in range(len(embedding_p))]
                    # update embeddings
                    embeddings = temp_embed
                    #print(f"middle loop {i+1} of {n}: {embeddings[0]}")

            # now average the embedding
            embeddings = [(embeddings[j]/n) for j in range(len(embeddings))]

        return embeddings

    # set paths
    depression_path = 'Depression'+suffix
    control_path = 'Control'+suffix

    # get list of text files from the depression folder and embed them
    text_files = [f for f in os.listdir(depression_path) if f.endswith(".txt")]

    # initialize the lists
    model1 = []
    model2 = []
    model3 = []
    model4 = []

    # set the embedding models
    transformer1 = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    transformer2 = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    transformer3 = SentenceTransformer('sentence-transformers/gtr-t5-base')
    transformer4 = SentenceTransformer('sentence-transformers/sentence-t5-base')

    # loop through the text files
    for file in tqdm(range(len(text_files)), desc="Implementing Sentence Transformers on Depression Posts"):
        #print(f"Working on document {file+1}/{len(text_files)}")

        # open the text file
        with open(os.path.join(depression_path, text_files[file]), "r", encoding = 'utf-8') as f:
            text = f.read()

        # embed the text and add to list
        model1.append(create_embeddings(text, transformer1))
        model2.append(create_embeddings(text, transformer2))
        model3.append(create_embeddings(text, transformer3))
        model4.append(create_embeddings(text, transformer4))

    # Create the pandas DataFrame
    df1 = pd.DataFrame(model1)
    df2 = pd.DataFrame(model2)
    df3 = pd.DataFrame(model3)
    df4 = pd.DataFrame(model4)

    # add Label "1" since the depression texts are the target texts
    df1["Label"] = 1
    df2["Label"] = 1
    df3["Label"] = 1
    df4["Label"] = 1

    # get list of text files from the control folder and embed them
    text_files = [f for f in os.listdir(control_path) if f.endswith(".txt")]

    # initialize the lists
    model1_c = []
    model2_c = []
    model3_c = []
    model4_c = []

    # loop through the text files
    for file in tqdm(range(len(text_files)), desc="Implementing Sentence Transformers on Control Posts"):

        # open the text file
        with open(os.path.join(control_path, text_files[file]), "r", encoding = 'utf-8') as f:
            text = f.read()

        # embed the text and add to list
        model1_c.append(create_embeddings(text, transformer1))
        model2_c.append(create_embeddings(text, transformer2))
        model3_c.append(create_embeddings(text, transformer3))
        model4_c.append(create_embeddings(text, transformer4))

    # Create the pandas DataFrame
    df1_c = pd.DataFrame(model1_c)
    df2_c = pd.DataFrame(model2_c)
    df3_c = pd.DataFrame(model3_c)
    df4_c = pd.DataFrame(model4_c)

    # add Label "0" since this is for the control texts
    df1_c["Label"] = 0
    df2_c["Label"] = 0
    df3_c["Label"] = 0
    df4_c["Label"] = 0

    # combine the dataframes
    df_model1 = pd.concat([df1, df1_c])
    df_model2 = pd.concat([df2, df2_c])
    df_model3 = pd.concat([df3, df3_c])
    df_model4 = pd.concat([df4, df4_c])

    
    # Create the output folder if it doesn't exist
    if not os.path.exists('transformer'):
        os.makedirs('transformer')

    # download dataframes as csv files
    df_model1.to_csv('transformer/transformer_all_minilm_l6_v2.csv', index=False)
    df_model2.to_csv('transformer/transformer_all_mpnet_base_v2.csv', index=False)
    df_model3.to_csv('transformer/transformer_gtr_t5_base.csv', index=False)
    df_model4.to_csv('transformer/transformer_sentence_t5_base.csv', index=False)
        
# transformer('_Clean')



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

    folders = ['word_embeddings', 'transformer']
    for folder in folders:
        process_files(folder, 'results.csv')

get_results()



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
    df.to_csv(results_file, index=False)
    print(f"Results saved as {modified_file}")

format_results('results.csv', 'final_results.csv')