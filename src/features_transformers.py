
# Sentence-transformer feature extraction
# Reads data/Depression<suffix> and data/Control<suffix> and writes 4 feature CSVs
# (all-MiniLM-L6-v2, all-mpnet-base-v2, gtr-t5-base, sentence-t5-base).
# suffix = '_Clean' uses the keyword-cleaned dataset (as in the published study)
# and writes to features/transformer/;
# suffix = '' uses the original posts and writes to features/transformer_unclean/
# (used by the keyword-removal comparison study, src/cross_domain.py).
#
# Run from the repository root:  python src/features_transformers.py
# (pass --original to embed the original, uncleaned posts instead)

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
    depression_path = 'data/Depression'+suffix
    control_path = 'data/Control'+suffix

    # Output folder: the cleaned dataset is the published one and goes to
    # features/transformer; the original posts go to a separate folder
    output_folder = 'features/transformer' if suffix == '_Clean' else 'features/transformer_unclean'

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
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # download dataframes as csv files
    df_model1.to_csv(f'{output_folder}/transformer_all_minilm_l6_v2.csv', index=False)
    df_model2.to_csv(f'{output_folder}/transformer_all_mpnet_base_v2.csv', index=False)
    df_model3.to_csv(f'{output_folder}/transformer_gtr_t5_base.csv', index=False)
    df_model4.to_csv(f'{output_folder}/transformer_sentence_t5_base.csv', index=False)


if __name__ == "__main__":
    import sys
    transformer('' if '--original' in sys.argv else '_Clean')
