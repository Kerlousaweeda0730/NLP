import os
import json

# We're using bag of words features. So for every review:
# We'll use a dictionary to keep track of words and their counts
# Also need to keep track of the vocabulary of a class
# Also need to keep track of the total number of tokens in a class


class Preprocess:

    small_dataset_path = 'small-corpus-reviews/'

    movie_review_path = 'movie-review-HW2/'

    def __init__(self):
        self.process_small_dataset()
        self.process_large_dataset()

    def validate_token(self, token):
        # Separate punctuation and lowercase words
        token = token.lower()
        token = token.replace(".", "")
        token = token.replace(",", "")
        return token

    # Process the smaller training corpus to make sure our preprocessing algorithm is working
    def process_small_dataset(self):
        training = self.small_dataset_path + 'aclImdb/train'
        test = self.small_dataset_path + 'aclImdb/test'
        output_path = self.small_dataset_path + 'feature_vectors'
        vocab = set()
        self.process_folder(training, vocab, output_path)
        self.process_folder(test, vocab, output_path)

    # Process the larger training corpus to use on our Naive Bayes model
    def process_large_dataset(self):
        training = self.movie_review_path + 'aclImdb/train'
        test = self.movie_review_path + 'aclImdb/test'
        output_path = os.path.join(self.movie_review_path, 'feature_vectors')
        # Use the imdb.vocab file we have
        vocab = set()
        # Add the words in the vocab file to the vocab set
        with open(os.path.join(self.movie_review_path, 'aclImdb/imdb.vocab'), 'r', encoding='utf8') as f:
            words = f.read().split()
            for word in words:
                vocab.add(word)
        self.process_folder(training, vocab, output_path)
        self.process_folder(test, vocab, output_path)

    def process_folder(self, path, vocab, output_path):

        # A list of vector representations to be output
        vectors = []

        # We're going to walk down the path given
        for dir, sub, files in os.walk(path):
            # If there are no more subfolders then continue to the files
            if len(sub) == 0:
                # Get the current class from the dir path, action or comedy
                curr_class = os.path.basename(dir)
                # Loop through the files
                for file_name in files:
                    file_path = os.path.join(dir, file_name)
                    # Open the file you're at
                    with open(file_path, 'r', encoding='utf8') as f:
                        # The current review's vector
                        # Format={ action : {
                        #                   word : count
                        #        }}
                        #curr_vector = {curr_class: {}}
                        curr_vector = [curr_class, {}]
                        # Line by line
                        for line in f:
                            tokens = line.split()
                            # Word by word
                            for token in tokens:
                                token = self.validate_token(token)
                                # If the token is in the vocabulary OR if the vocabulary is empty
                                if token not in vocab and len(vocab) > 0:
                                    continue
                                # Keep track of the count of the word
                                # try:
                                #     curr_vector[curr_class][token] += 1
                                # except KeyError:
                                #     curr_vector[curr_class][token] = 1
                                try:
                                    curr_vector[1][token] += 1
                                except KeyError:
                                    curr_vector[1][token] = 1
                    vectors.append(curr_vector)

        # Add train or test to the output path
        file_name = os.path.basename(path) + '_output_vector.json'
        output_path = os.path.join(output_path, os.path.basename(path))
        # Make the folders if they dont exist
        os.makedirs(output_path, exist_ok=True)
        # Write to the file the vector output
        with open(os.path.join(output_path, file_name), 'w', encoding='utf8') as output:
            output.write(json.dumps(vectors, indent=2))


p = Preprocess()
