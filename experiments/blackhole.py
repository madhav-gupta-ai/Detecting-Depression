
# Master File

def dataset_collection():
    
        
    # Detecting Depression in Text using Word Embeddings
    
    # Madhav Gupta
    # Prof. V. K. Jayaraman
    # B. Sc. Honours Thesis
    # FLAME University
    # 28th April, 2023
    
    # Program 1: Dataset Collection using Reddit API
    
    
    # Libraries used:
    
    import praw     # the official Reddit API used
    import hashlib  # provides the SHA-1 hash encoding used for user anonymity
    from prawcore.exceptions import Forbidden   # PRAW exception, to be handled
    
    from datetime import datetime, timedelta    # used to check date and timestamps
    from statistics import mean, median         # functions to compute averages
    
        
    # Some dataset specifications
    
    n_users = 1000          # no. of users to consider at a time (PRAW max: 1000)
    n_posts_per_user = 5    # max. no. of posts to take from a single author
    n_recent_posts = None   # how far back in user history (None = the max 1000 posts)
    n_days_post_diag = 365  # how many days of posts to take after a diagnosis
    n_chars_per_post = 60   # min. no. of chars per post, 6 chars ~ 1 word + 1 punctuation
    
    
    # Extracting some lists of phrases from .txt files
    
    print("Initializing...\n")
    
    # List of Mental Health Subreddits of interest
    with open('mh_subreddits.txt') as file:
        # Read each line and store it in a list
        mh_subreddits = [line.strip().lower() for line in file]
    
    # List of search queries related to Diagnosis Posts   
    with open('diagnosis_words.txt') as file:
        search_queries = [line.strip().lower() for line in file]
    
    # List of Diagnosis Post phrases
    with open('positive_diagnosis.txt') as file:
        positive_diagnosis = [line.strip().lower() for line in file]
    
    # List of false-positive Diagnosis Post phrases
    with open('negative_diagnosis.txt') as file:
        negative_diagnosis = [line.strip().lower() for line in file]
    
    # List of phrases referring to Clinical Depression
    with open('condition_words.txt') as file:
        condition_words = [line.strip().lower() for line in file]
    
    # List of phrases related to Mental Health
    with open('mh_patterns.txt') as file:
        mh_patterns = [line.strip().lower() for line in file]
    
    
    # Initializing dataset statistics
        
    depression_posts_count = 1
    control_posts_count = 1
    user_count = 1
    total_depression_posts_count = []
    depression_posts_wordcounts = []
    
    
    # Loading the Reddit API Credentials
    
    with open('API_credentials.txt') as file:
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
    
    
    # Searching and obtaining the search results for some search queries related 
    # to Diagnosis Posts from r/depression - to find possibly diagnosed users
    
    subreddit = reddit.subreddit('depression')
    print("Obtaining posts from r/depression")
    for search_query in search_queries:
        print(f"Searching {search_query}")
        for hot_post in subreddit.search(search_query, sort='relevance', limit = n_users):
            
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
                
            for post in user_posts:
                depression_posts_wordcounts += [len(post[3])]
    
        
            # Saving the post title and body text as .txt files
    
            print(f"Found {len(user_posts)} acceptable posts\n")
    
            for i in range(len(user_posts)):
                
                filename = f"Depression/post_{depression_posts_count}.txt"
                depression_posts_count+=1
                
                with open(filename, 'w', encoding='utf-8') as file:
                    file.write(user_posts[i][0] + '\n')
                    file.write(user_posts[i][3])
                    
            
    
    # Get hot (trending) posts from r/CasualConversation (control group)
    
    subreddit = reddit.subreddit('casualconversation')
    print("Obtaining control posts from r/casualconversation")
    for hot_post in subreddit.hot(limit = n_users):
        
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
        
    
        # Saving the post title and body text as .txt files
    
        print("Saving the post")
        filename = f"Control/post_{control_posts_count}.txt"
        control_posts_count+=1
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(hot_post.title + '\n')
            file.write(hot_post.selftext)    
    
            
    # Printing the depression posts findings and statistics        
    
    print("\nTotal no. of posts found:", sum(total_depression_posts_count))
    print("Mean posts per user:", mean(total_depression_posts_count))
    print("Median number of posts:", median(total_depression_posts_count))        
    print("Mean length of posts:", mean(depression_posts_wordcounts), "chars")
    print("Median length of posts:", median(depression_posts_wordcounts), "chars")        
    
    # Open the file in write mode
    file_path = 'dataset_stats.txt'  # Specify the file path
    with open(file_path, 'w') as file:
        # Write content to the file
        file.write(f"Total no. of posts found: {sum(total_depression_posts_count)}\n")
        file.write(f"Mean posts per user: {mean(total_depression_posts_count)}\n")
        file.write(f"Median number of posts: {median(total_depression_posts_count)}\n")
        file.write(f"Mean length of posts: {mean(depression_posts_wordcounts)} chars\n")
        file.write(f"Median length of posts: {median(depression_posts_wordcounts)} chars\n")
    
    

#dataset_collection()




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
            with open(words_file, 'r', encoding = 'utf-8') as f:
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




#remove_words()




def word_embedding(folder, csv_version):
    
    
    
    # Detecting Depression in Text using Word Embeddings
    
    # Madhav Gupta
    # Prof. V. K. Jayaraman
    # B. Sc. Honours Thesis
    # FLAME University
    # 28th April, 2023
    
    # Program 2: Data Processing and Word-Embeddings Generation 
        
    
    # Libraries used:
                        
    import os           # Used to import and traverse the system's directories
    import csv          # Used to read and write to a .csv file
    import re           # Used to implement Regular Expressions
    import numpy as np  # Working with arrays at a high level
    import gensim.downloader as api     # Loads the GloVe word embedding models
    from nltk.stem import WordNetLemmatizer     # Used during pre-processing
    
    
    # Define the path to the folders containing the text files
    text_folder_path = 'Depression'+folder
    control_folder_path = 'Control'+folder
    
    
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
        filename = f'word_embeddings_{n_dimensions}_{sum_or_average}{csv_version}_imbalanced.csv'
        
        print(filename)
        
        # Loading the GloVe model
        model = api.load(modelname)
        
        with open(filename, "w", newline="", encoding = 'utf-8') as csv_file:
            writer = csv.writer(csv_file)
            
            for text_file in text_files:
                # Finding the embeddings of depression posts and saving
                embedding = findEmbedding(model, n_dimensions, text_file, text_folder_path, sum_or_average)
                writer.writerow(embedding + ['1'])  # class 1: depression
    
            for text_file in control_files:
                # Finding the embeddings of control posts and saving
                embedding = findEmbedding(model, n_dimensions, text_file, control_folder_path, sum_or_average)
                writer.writerow(embedding + ['0'])  # class 0: control
            
    
    # Making seperate word embedding feature extractions for
    # 5 different GloVe Twitter models: 25, 50, 100 and 200 dimensions, and for each 
    # on the basis of calculation by sum and calculation by average word vectors
    
    for modelname in [f'glove-twitter-{x}' for x in [25, 50, 100, 200]]:
        for sum_or_average in ['sum', 'average']:
            
            print(f"GloVe {modelname[14:]}-D Twitter Word Embeddings Model in use")        
            makeCSV(modelname, sum_or_average)
    
        
    
#word_embedding('','')    
#word_embedding('_Clean','_clean')





def balancing():
    

    # Performing SMOTE Class Balancing
    
    import pandas as pd
    from imblearn.over_sampling import SMOTE
    
    for n_dimensions in [25, 50, 100, 200]:
                
        for sum_or_average in ['sum', 'average']:
            
            for clean in ['', '_clean']:
                        
                filename = f'word_embeddings_{n_dimensions}_{sum_or_average}{clean}'
                
                print(filename)
                
                data = pd.read_csv(filename+'_imbalanced.csv')
        
                # Separate the attributes and the class labels
                X = data.iloc[:, :-1]  # Features/attributes
                y = data.iloc[:, -1]   # Class labels
                
                # Apply SMOTE for class balancing
                smote = SMOTE()
                X_balanced, y_balanced = smote.fit_resample(X, y)
                
                # Create a new balanced DataFrame
                balanced_data = pd.concat([X_balanced, y_balanced], axis=1)
                
                # Save the balanced data to a new CSV file
                balanced_data.to_csv(filename+'.csv', index=False)
        
    
#balancing()











import numpy as np


# Blackhole hyperparameters
num_stars = 8
num_iter = 10
min_improvement = 0.0001




print("Begin Testing")

def fulltesting(csv_version):
    

    # Full Testing
    
    
    
    def logreg():
        
        
        # Detecting Depression in Text using Word Embeddings
        
        # Madhav Gupta
        # Prof. V. K. Jayaraman
        # B. Sc. Honours Thesis
        # FLAME University
        # 28th April, 2023
        
        # Program 3: Logistic Regression Model
        
        
        attempts = 5
        
        import csv
        import pandas as pd
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split, GridSearchCV
        from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score, make_scorer
        from statistics import mode, mean
        
        import warnings
        from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        

        def model(X,y):
            
            # Create a Logistic Regression model
            logistic_model = LogisticRegression()
            
            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            
            # Define the parameter grid for grid search
            # param_grid = {
            #     'C': [0.1, 1, 10, 100],
            #     'penalty': ['l1', 'l2'],
            #     'solver': ['liblinear', 'saga']
            # }
            
            param_grid = {
                'penalty': ['l1', 'l2'],  # Regularization penalty to apply
                'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization strength
                'solver': ['liblinear', 'saga'],  # Algorithm to use in the optimization problem
                #'class_weight': [None, 'balanced'],  # Weights associated with classes
                'max_iter': [1000],  # Maximum number of iterations to converge
                #'threshold': [0.3, 0.4, 0.5, 0.6, 0.7]  # Threshold for classification
            }
            
            # Define custom scorer for F1 score on positive class only
            scorer = make_scorer(f1_score, pos_label=1)
            
            # Create a GridSearchCV object
            grid_search = GridSearchCV(logistic_model, param_grid, cv=5, scoring=scorer, n_jobs=-1, verbose=1)
            
            # Fit the GridSearchCV instance to the training data
            grid_search.fit(X_train, y_train)
            
            # Print the best parameters found by grid search
            print('Best Parameters:', grid_search.best_params_)
            print("Best score:", grid_search.best_score_)
            
            # Create and test a model with the best parameters
            best_model = LogisticRegression(**grid_search.best_params_)
            best_model.fit(X_train, y_train)
            y_pred = best_model.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            print("Accuracy:", acc)
            
            f1 = f1_score(y_test, y_pred, pos_label=1)
            print("F1 Score:", f1)
            
            rep = classification_report(y_test, y_pred)
            print('Classification Report:')
            print(rep)
            
            conf = confusion_matrix(y_test, y_pred)
            print('Confusion Matrix:')
            print(conf)
            
            
            return (grid_search.best_params_, acc, f1, conf)
        


        def test(filename, threshold):
        
            # Load the data from the CSV file
            df = pd.read_csv(filename)
            
            # Drop the second last column and store the target labels in a separate variable
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]


            # Perform BBHA



            num_features = X.shape[1]

            def initialize_stars(num_stars, num_features):
                stars = np.random.randint(2, size=(num_stars, num_features))
                # Set all values of the first row to 1
                stars[0, :] = 1
                # Make the second star half 1, half 0
                stars[1, :num_features//2] = 1
                stars[1, num_features//2:] = 0
                # Make the third star half 0, half 1
                stars[2, :num_features//2] = 0
                stars[2, num_features//2:] = 1        
                return stars
            
            
            def fitness(star, X, y):
                selected_features = X.columns[star == 1]
                if len(selected_features) == 0:
                    return 0
                X_selected = X[selected_features]
                return model(X_selected, y)[2]
            
            def update_position(stars, black_hole, R):
                for i in range(len(stars)):
                    distance = np.sqrt(np.sum((black_hole - stars[i]) ** 2))
                    if distance < R:
                        stars[i] = np.random.randint(2, size=num_features)  # Replace with a new star
                    else:
                        for d in range(num_features):
                            stars[i, d] += np.random.rand() * (black_hole[d] - stars[i, d])
                            stars[i, d] = 1 if abs(np.tanh(stars[i, d])) > threshold else 0
                return stars
            
            stars = initialize_stars(num_stars, num_features)
            fitness_values = np.array([fitness(star, X, y) for star in stars])

            #black_hole_index = np.argmax(fitness_values)

            # Find the index of the highest value
            max_index = np.argmax(fitness_values)

            # Check if the selected black hole is an array of all 1's
            if np.all(stars[max_index] == 1):
                # Set the value at the max_index to a very low value
                max_value = fitness_values[max_index]
                fitness_values[max_index] = 0

                # Find the index of the second highest value
                second_max_index = np.argmax(fitness_values)

                # Use the second highest index as the black hole index
                black_hole_index = second_max_index

                fitness_values[max_index] = max_value

            else:
                # Use the original black hole index
                black_hole_index = max_index

            black_hole = stars[black_hole_index]
            best_fitness = 0
            
            iter = 0
            while iter<num_iter and fitness_values[black_hole_index] >= best_fitness + min_improvement:

                print("Blackhole Iteration", iter)

                best_fitness = fitness_values[black_hole_index]

                R = fitness_values[black_hole_index] / np.sum(fitness_values)

                stars = update_position(stars, black_hole, R)
                new_fitness_values = np.array([fitness(star, X, y) for star in stars])

                # Check if a star has a better fitness than the black hole
                for i, fitness_val in enumerate(new_fitness_values):
                    if ((fitness_val > fitness_values[black_hole_index]) or ((fitness_val == fitness_values[black_hole_index]) and (np.sum(stars[i]) < np.sum(stars[black_hole_index])))) and (not np.all(stars[i] == 1)):
                        black_hole = stars[i]
                        black_hole_index = i

                fitness_values = new_fitness_values


            print("Best black hole:", black_hole)
            return model(X[X.columns[black_hole == 1]], y), black_hole



            
            #return (grid_search.best_params_, acc, f1, conf)
            
            
        output_file = f'results_logreg{csv_version}.csv'
        
        # Open the file in append mode
        with open(output_file, 'a', newline='') as out:
            # Create a CSV writer object
            writer = csv.writer(out)
            
            writer.writerow(['Attempt',
                             'dimensions', 
                             'method', 
                             'C', 
                             #'class_weight', 
                             'max_iter', 
                             'penalty', 
                             'solver', 
                             'tp',
                             'fn',
                             'fp',
                             'tn',
                             'accuracy', 
                             'f1_score', 
                             'blackhole_threshold',
                             'black_hole'
                             ])    
            
            
            results = []
            for n_dimensions in [25, 50, 100, 200]:
                
                for sum_or_average in ['sum', 'average']:
                    
                    filename = f'word_embeddings_{n_dimensions}_{sum_or_average}{csv_version}.csv'
                    
                    print("\n", filename)
                    
                    attempts = []
                    
                    for threshold in [0.4, 0.5, 0.6]:
                        
                        attempt_number = 1

                        results, black_hole = test(filename, threshold)
                        
                        testsize = sum([item for sublist in results[3] for item in sublist])
                        
                        attempt = [attempt_number,
                                   n_dimensions,
                                   sum_or_average,
                                   results[0]['C'], 
                                   #results[0]['class_weight'], 
                                   results[0]['max_iter'], 
                                   results[0]['penalty'], 
                                   results[0]['solver'], 
                                   results[3][1][1] / testsize, #tp
                                   results[3][1][0] / testsize, #fn
                                   results[3][0][1] / testsize, #fp
                                   results[3][0][0] / testsize, #tn
                                   results[1], # accuracy
                                   results[2],  # f1-score
                                   threshold,
                                   black_hole
                                   ]
                        
                        attempts += [attempt]
                        
                        # Write the data to the file
                        writer.writerow(attempt)
                        
                    
                    #attempts_average = ['Avg'] + [mode([sublist[i] for sublist in attempts]) for i in range(1,8)] + [mean([sublist[i] for sublist in attempts]) for i in range(8, 14)]
                    attempts_average = ['Avg'] + [mode([sublist[i] for sublist in attempts]) for i in range(1,7)] + [mean([sublist[i] for sublist in attempts]) for i in range(7, 13)]
                    
                    writer.writerow(attempts_average)
                    
                    
    #logreg()                
                    
    
    
    
    svm_fitnesses = {}
    
    def svm():
        
        
        # Detecting Depression in Text using Word Embeddings
        
        # Madhav Gupta
        # Prof. V. K. Jayaraman
        # B. Sc. Honours Thesis
        # FLAME University
        # 28th April, 2023
        
        # Program 4: Support Vector Machines Model Test
        
        
        import csv
        import pandas as pd
        from sklearn.svm import SVC
        from sklearn.model_selection import train_test_split, GridSearchCV
        from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score, make_scorer
        from statistics import mode, mean
        
        import warnings
        from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        
        
        def model(X,y, stari):
            
            star = stari.tostring()

            if star in svm_fitnesses.keys():
                return svm_fitnesses[star]

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            
            # Set up the parameter grid for GridSearchCV
            # param_grid = {
            #     "C": [0.1, 1, 10, 100],
            #     "kernel": ["linear", "rbf", "sigmoid", "poly"],
            #     "degree": [2,3,4],
            #     "gamma": ["scale", "auto"]
            # }
            
            param_grid = {
                'kernel': ['rbf', 'sigmoid', 'poly'],
                'C': [0.01, 0.1, 1, 10, 100],
                'gamma': ['scale', 'auto'] + [0.1, 1],
                #'class_weight': ['balanced', None],
                'probability': [True],
                'decision_function_shape': ['ovr'],
                #'decision_function_threshold': [0.3, 0.4, 0.5, 0.6, 0.7]  
            }
        
            
            # Create an SVM instance
            svm = SVC()
            
            # Define custom scorer for F1 score on positive class only
            scorer = make_scorer(f1_score, pos_label=1)
            print("starting gridsearch")
            # Create a GridSearchCV instance
            grid_search = GridSearchCV(svm, param_grid, cv=5, scoring=scorer, n_jobs=-1, verbose=1)
            print("starting gridsearch now")
            # Fit the GridSearchCV instance to the training data
            grid_search.fit(X_train, y_train)
            
            # Print the best parameters found by grid search
            print('Best Parameters:', grid_search.best_params_)
            print("Best score:", grid_search.best_score_)
            
            # Create and test a model with the best parameters
            best_model = SVC(**grid_search.best_params_)
            best_model.fit(X_train, y_train)
            y_pred = best_model.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            print("Accuracy:", acc)
            
            f1 = f1_score(y_test, y_pred, pos_label=1)
            print("F1 Score:", f1)
            
            rep = classification_report(y_test, y_pred)
            print('Classification Report:')
            print(rep)
            
            conf = confusion_matrix(y_test, y_pred)
            print('Confusion Matrix:')
            print(conf)

            svm_fitnesses[star] = (grid_search.best_params_, acc, f1, conf)
            
            return (grid_search.best_params_, acc, f1, conf)



        def test(filename, threshold):
        
            # Load the CSV file
            data = pd.read_csv(filename)
            
            # Split the data into features (X) and target label (y)
            X = data.iloc[:, :-1] # drop the second last column
            y = data.iloc[:, -1]
            
            # Perform BBHA



            num_features = X.shape[1]

            def initialize_stars(num_stars, num_features):
                stars = np.random.randint(2, size=(num_stars, num_features))
                # Set all values of the first row to 1
                stars[0, :] = 1
                # Make the second star half 1, half 0
                stars[1, :num_features//2] = 1
                stars[1, num_features//2:] = 0
                # Make the third star half 0, half 1
                stars[2, :num_features//2] = 0
                stars[2, num_features//2:] = 1        
                return stars
            
            def fitness(star, X, y):
                selected_features = X.columns[star == 1]
                if len(selected_features) == 0:
                    return 0
                X_selected = X[selected_features]
                return model(X_selected, y, star)[2]
            
            def update_position(stars, black_hole, R):
                for i in range(len(stars)):
                    distance = np.sqrt(np.sum((black_hole - stars[i]) ** 2))
                    if distance < R:
                        stars[i] = np.random.randint(2, size=num_features)  # Replace with a new star
                    else:
                        for d in range(num_features):
                            stars[i, d] += np.random.rand() * (black_hole[d] - stars[i, d])
                            stars[i, d] = 1 if abs(np.tanh(stars[i, d])) > threshold else 0
                return stars
            
            stars = initialize_stars(num_stars, num_features)
            fitness_values = np.array([fitness(star, X, y) for star in stars])

            #black_hole_index = np.argmax(fitness_values)

            # Find the index of the highest value
            max_index = np.argmax(fitness_values)

            # Check if the selected black hole is an array of all 1's
            if np.all(stars[max_index] == 1):
                # Set the value at the max_index to a very low value
                max_value = fitness_values[max_index]
                fitness_values[max_index] = 0

                # Find the index of the second highest value
                second_max_index = np.argmax(fitness_values)

                # Use the second highest index as the black hole index
                black_hole_index = second_max_index

                fitness_values[max_index] = max_value

            else:
                # Use the original black hole index
                black_hole_index = max_index

            black_hole = stars[black_hole_index]
            best_fitness = 0
            
            iter = 0
            while iter<num_iter and fitness_values[black_hole_index] >= best_fitness + min_improvement:

                print("Blackhole Iteration", iter)

                best_fitness = fitness_values[black_hole_index]

                R = fitness_values[black_hole_index] / np.sum(fitness_values)

                stars = update_position(stars, black_hole, R)
                new_fitness_values = np.array([fitness(star, X, y) for star in stars])

                # Check if a star has a better fitness than the black hole
                for i, fitness_val in enumerate(new_fitness_values):
                    if ((fitness_val > fitness_values[black_hole_index]) or ((fitness_val == fitness_values[black_hole_index]) and (np.sum(stars[i]) < np.sum(stars[black_hole_index])))) and (not np.all(stars[i] == 1)):
                        black_hole = stars[i]
                        black_hole_index = i

                fitness_values = new_fitness_values


            print("Best black hole:", black_hole)
            return model(X[X.columns[black_hole == 1]], y, black_hole), black_hole


        
        
        
        output_file = f'results_svm{csv_version}.csv'
        
        # Open the file in append mode
        with open(output_file, 'a', newline='') as out:
            # Create a CSV writer object
            writer = csv.writer(out)
            
            writer.writerow(['Attempt',
                             'dimensions', 
                             'method', 
                             'kernel', 
                             'C', 
                             'gamma', 
                             #'class_weight', 
                             'probability', 
                             'decision_function_shape',
                             'tp',
                             'fn',
                             'fp',
                             'tn',
                             'accuracy', 
                             'f1_score', 
                             'blackhole_threshold',
                             'black_hole'
                             ])    
            
            
            results = []
            for n_dimensions in [200, 100, 50, 25]:
                
                for sum_or_average in ['sum', 'average']:
                    
                    filename = f'word_embeddings_{n_dimensions}_{sum_or_average}{csv_version}.csv'
                    
                    print("\n", filename)
                    
                    attempts = []
                    
                    for threshold in [0.4,0.5,0.6]:

                        attempt_number = 1
                    
                        results, black_hole = test(filename, threshold)
                        
                        testsize = sum([item for sublist in results[3] for item in sublist])
                        
                        attempt = [attempt_number,
                                   n_dimensions,
                                   sum_or_average,
                                   results[0]['kernel'], 
                                   results[0]['C'], 
                                   results[0]['gamma'], 
                                   #results[0]['class_weight'], 
                                   results[0]['probability'],
                                   results[0]['decision_function_shape'],
                                   results[3][1][1] / testsize, #tp
                                   results[3][1][0] / testsize, #fn
                                   results[3][0][1] / testsize, #fp
                                   results[3][0][0] / testsize, #tn
                                   results[1], # accuracy
                                   results[2],  # f1-score
                                   threshold,
                                   black_hole
                                   ]
                        
                        attempts += [attempt]
                        
                        # Write the data to the file
                        writer.writerow(attempt)
                        
                    
                    #attempts_average = ['Avg'] + [mode([sublist[i] for sublist in attempts]) for i in range(1,9)] + [mean([sublist[i] for sublist in attempts]) for i in range(9, 15)]
                    attempts_average = ['Avg'] + [mode([sublist[i] for sublist in attempts]) for i in range(1,8)] + [mean([sublist[i] for sublist in attempts]) for i in range(8, 14)]
                    
                    writer.writerow(attempts_average)
                    
                    
                    
    #svm()
    
    
    
    
    
    def randforest():
        
        
        # Detecting Depression in Text using Word Embeddings
        
        # Madhav Gupta
        # Prof. V. K. Jayaraman
        # B. Sc. Honours Thesis
        # FLAME University
        # 28th April, 2023
        
        # Program 4: Random Forest Model Test
        
        
        import csv
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split, GridSearchCV
        from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score, make_scorer
        
        import warnings
        from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        
            
        def model(X,y):

            # Create a Random Forest Classifier
            rfc_model = RandomForestClassifier()
            
            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            
            # Define the parameter grid for grid search
            # param_grid = {
            #     'n_estimators': [50, 100, 200],
            #     'criterion': ['gini', 'entropy'],
            #     'max_depth': [None, 10, 50],
            #     'min_samples_split': [2, 5, 10],
            #     'min_samples_leaf': [1, 2, 4],
            # }
            
            param_grid = {
                'n_estimators': [25, 50, 100, 150],  # Number of trees in the forest
                'max_features': ['sqrt', 'log2'],  # Maximum number of features to consider for each split
                'max_depth': [None, 10, 15],  # Maximum depth of the tree
                'min_samples_split': [2, 5],  # Minimum number of samples required to split an internal node
                'min_samples_leaf': [1, 2],  # Minimum number of samples required to be at a leaf node
                #'class_weight': [None, 'balanced'],  # Weights associated with classes
                'criterion': ['gini', 'entropy'],  # Function to measure the quality of a split
                'bootstrap': [True],  # Whether bootstrap samples are used when building trees
                #'threshold': [0.3, 0.4, 0.5, 0.6, 0.7]  # Threshold for classification
            }
            
            # Define custom scorer for F1 score on positive class only
            scorer = make_scorer(f1_score, pos_label=1)
            
            # Create a GridSearchCV object
            grid_search = GridSearchCV(rfc_model, param_grid, cv=5, scoring=scorer, n_jobs=-1, verbose=3)
            
            # Fit the GridSearchCV instance to the training data
            grid_search.fit(X_train, y_train)
            
            # Print the best parameters found by grid search
            print('Best Parameters:', grid_search.best_params_)
            print("Best score:", grid_search.best_score_)
            
            # Create and test a model with the best parameters
            best_model = RandomForestClassifier(**grid_search.best_params_)
            best_model.fit(X_train, y_train)
            y_pred = best_model.predict(X_test)
        
            
            acc = accuracy_score(y_test, y_pred)
            print("Accuracy:", acc)
            
            f1 = f1_score(y_test, y_pred, pos_label=1)
            print("F1 Score:", f1)
            
            rep = classification_report(y_test, y_pred)
            print('Classification Report:')
            print(rep)
            
            conf = confusion_matrix(y_test, y_pred)
            print('Confusion Matrix:')
            print(conf)
            
            return (grid_search.best_params_, acc, f1, conf)
        


        def test(filename, threshold):
        
            # Load the data from the CSV file
            df = pd.read_csv(filename)
            
            # Drop the second last column and store the target labels in a separate variable
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
            
            # Perform BBHA



            num_features = X.shape[1]

            def initialize_stars(num_stars, num_features):
                stars = np.random.randint(2, size=(num_stars, num_features))
                # Set all values of the first row to 1
                stars[0, :] = 1
                # Make the second star half 1, half 0
                stars[1, :num_features//2] = 1
                stars[1, num_features//2:] = 0
                # Make the third star half 0, half 1
                stars[2, :num_features//2] = 0
                stars[2, num_features//2:] = 1        
                return stars
            
            def fitness(star, X, y):
                selected_features = X.columns[star == 1]
                if len(selected_features) == 0:
                    return 0
                X_selected = X[selected_features]
                return model(X_selected, y)[2]
            
            def update_position(stars, black_hole, R):
                for i in range(len(stars)):
                    distance = np.sqrt(np.sum((black_hole - stars[i]) ** 2))
                    if distance < R:
                        stars[i] = np.random.randint(2, size=num_features)  # Replace with a new star
                    else:
                        for d in range(num_features):
                            stars[i, d] += np.random.rand() * (black_hole[d] - stars[i, d])
                            stars[i, d] = 1 if abs(np.tanh(stars[i, d])) > threshold else 0
                return stars
            
            stars = initialize_stars(num_stars, num_features)
            fitness_values = np.array([fitness(star, X, y) for star in stars])

            #black_hole_index = np.argmax(fitness_values)

            # Find the index of the highest value
            max_index = np.argmax(fitness_values)

            # Check if the selected black hole is an array of all 1's
            if np.all(stars[max_index] == 1):
                # Set the value at the max_index to a very low value
                max_value = fitness_values[max_index]
                fitness_values[max_index] = 0

                # Find the index of the second highest value
                second_max_index = np.argmax(fitness_values)

                # Use the second highest index as the black hole index
                black_hole_index = second_max_index

                fitness_values[max_index] = max_value

            else:
                # Use the original black hole index
                black_hole_index = max_index

            black_hole = stars[black_hole_index]
            best_fitness = 0
            
            iter = 0
            while iter<num_iter and fitness_values[black_hole_index] >= best_fitness + min_improvement:

                print("Blackhole Iteration", iter)

                best_fitness = fitness_values[black_hole_index]

                R = fitness_values[black_hole_index] / np.sum(fitness_values)

                stars = update_position(stars, black_hole, R)
                new_fitness_values = np.array([fitness(star, X, y) for star in stars])

                # Check if a star has a better fitness than the black hole
                for i, fitness_val in enumerate(new_fitness_values):
                    if ((fitness_val > fitness_values[black_hole_index]) or ((fitness_val == fitness_values[black_hole_index]) and (np.sum(stars[i]) < np.sum(stars[black_hole_index])))) and (not np.all(stars[i] == 1)):
                        black_hole = stars[i]
                        black_hole_index = i

                fitness_values = new_fitness_values


            print("Best black hole:", black_hole)
            return model(X[X.columns[black_hole == 1]], y), black_hole
        
        
        
        output_file = f'results_randforest{csv_version}.csv'
        
        # Open the file in append mode
        with open(output_file, 'a', newline='') as out:
            # Create a CSV writer object
            writer = csv.writer(out)
            
            writer.writerow(['Attempt',
                             'dimensions', 
                             'method', 
                             'n_estimators', 
                             'max_features', 
                             'max_depth', 
                             'min_samples_split', 
                             'min_samples_leaf', 
                             #'class_weight',
                             'criterion',
                             'bootstrap',
                             'tp',
                             'fn',
                             'fp',
                             'tn',
                             'accuracy', 
                             'f1_score', 
                             'blackhole_threshold',
                             'black_hole'
                             ])    
            
            out.flush()
            
            results = []
            for n_dimensions in [200, 100, 50, 25]:
                
                for sum_or_average in ['sum', 'average']:
                    
                    filename = f'word_embeddings_{n_dimensions}_{sum_or_average}{csv_version}.csv'
                    
                    print("\n", filename)
                    
                    attempts = []
                    
                    for threshold in [0.4, 0.5, 0.6]:

                        attempt_number = 1
                    
                        results, black_hole = test(filename, threshold)
                        
                        testsize = sum([item for sublist in results[3] for item in sublist])
                        
                        attempt = [attempt_number,
                                   n_dimensions,
                                   sum_or_average,
                                   results[0]['n_estimators'], 
                                   results[0]['max_features'], 
                                   results[0]['max_depth'], 
                                   results[0]['min_samples_split'], 
                                   results[0]['min_samples_leaf'],
                                   #results[0]['class_weight'],
                                   results[0]['criterion'],
                                   results[0]['bootstrap'],
                                   results[3][1][1] / testsize, #tp
                                   results[3][1][0] / testsize, #fn
                                   results[3][0][1] / testsize, #fp
                                   results[3][0][0] / testsize, #tn
                                   results[1], # accuracy
                                   results[2],  # f1-score
                                   threshold,
                                   black_hole
                                   ]
                        
                        attempts += [attempt]
                        
                        # Write the data to the file
                        writer.writerow(attempt)
                        
                        out.flush()
                        
                    
                    #attempts_average = ['Avg'] + [mode([sublist[i] for sublist in attempts]) for i in range(1,11)] + [mean([sublist[i] for sublist in attempts]) for i in range(11, 17)]
                    
                    #writer.writerow(attempts_average)
                    
                    
                    
    #randforest()
    svm()
        
        
fulltesting('_clean')
#fulltesting('_clean_imbalanced')
#fulltesting('_imbalanced')
#fulltesting('')