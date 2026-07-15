
# Keyword cleaning ("cleaning" step of the study)
# Copies data/Depression -> data/Depression_Clean and data/Control -> data/Control_Clean
# with psychiatry-related keywords removed (condition, doctor and diagnosis words).
#
# Run from the repository root:  python src/clean.py

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
            with open(f'data/support/{words_file}', 'r', encoding = 'utf-8') as f:
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
    input_folder_depression = 'data/Depression'
    input_folder_control = 'data/Control'

    # Specify the output folder where the copies will be created
    output_folder_depression = 'data/Depression_Clean'
    output_folder_control = 'data/Control_Clean'

    # Specify the files containing the words to remove
    words_files = ['condition_words.txt', 'doctor_words.txt', 'diagnosis_words.txt']

    # Call the function to remove words in the depression folder
    remove_words_in_folder(input_folder_depression, output_folder_depression, words_files)

    # Call the function to remove words in the control folder
    remove_words_in_folder(input_folder_control, output_folder_control, words_files)


if __name__ == "__main__":
    remove_words()
