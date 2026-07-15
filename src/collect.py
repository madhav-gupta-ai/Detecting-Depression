
# Dataset collection using the Reddit API (PRAW)
# Collects Depression posts (r/depression, diagnosis-verified users) and
# Control posts (r/casualconversation) into data/Depression and data/Control.
#
# Requires data/support/API_credentials.txt (3 lines: client_id, client_secret,
# user_agent). This file is gitignored - keep your credentials out of the repo.
#
# Run from the repository root:  python src/collect.py

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
    with open('data/support/mh_subreddits.txt') as file:
        # Read each line and store it in a list
        mh_subreddits = [line.strip().lower() for line in file]

    # List of search queries related to Diagnosis Posts
    with open('data/support/diagnosis_words.txt') as file:
        search_queries = [line.strip().lower() for line in file]

    # List of Diagnosis Post phrases
    with open('data/support/positive_diagnosis.txt') as file:
        positive_diagnosis = [line.strip().lower() for line in file]

    # List of false-positive Diagnosis Post phrases
    with open('data/support/negative_diagnosis.txt') as file:
        negative_diagnosis = [line.strip().lower() for line in file]

    # List of phrases referring to Clinical Depression
    with open('data/support/condition_words.txt') as file:
        condition_words = [line.strip().lower() for line in file]

    # List of phrases related to Mental Health
    with open('data/support/mh_patterns.txt') as file:
        mh_patterns = [line.strip().lower() for line in file]


    # Create the Depression folder if it doesn't exist
    if not os.path.exists('data/Depression'):
        os.makedirs('data/Depression')

    # Create the Control folder if it doesn't exist
    if not os.path.exists('data/Control'):
        os.makedirs('data/Control')


    # Initializing dataset statistics
    depression_posts_count = len(os.listdir('data/Depression'))
    control_posts_count = len(os.listdir('data/Control'))
    user_count = 1
    total_depression_posts_count = []
    depression_posts_wordcounts = []


    # Loading the Reddit API Credentials
    with open('data/support/API_credentials.txt') as file:
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

    existing_depression_posts = get_existing_posts('data/Depression')
    existing_control_posts = get_existing_posts('data/Control')

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

                    filename = f"data/Depression/post_{depression_posts_count}.txt"
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
        filename = f"data/Control/post_{control_posts_count}.txt"
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(post_content)
        existing_control_posts.append(post_content)


    # Printing the depression posts findings and statistics
    print("\nTotal no. of posts found:", sum(total_depression_posts_count))
    print("Mean posts per user:", mean(total_depression_posts_count))
    print("Median number of posts:", median(total_depression_posts_count))
    print("Mean length of posts:", mean(depression_posts_wordcounts), "chars")
    print("Median length of posts:", median(depression_posts_wordcounts), "chars")


if __name__ == "__main__":
    dataset_collection()
