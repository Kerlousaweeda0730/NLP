import os
import json
import math


class NaiveBayes:
    # The small corpus
    small_dataset_path = 'small-corpus-reviews/'
    # The larger corpus
    movie_review_path = 'movie-review-HW2/'

    def __init__(self):
        self.train_model(self.small_dataset_path)

        self.train_model(self.movie_review_path)

    def train_model(self, dir):
        # To train our model we need:
        # - The prior probability of each class (count of reviews in that class / total reviews)
        # - The probability of each word given each class (count of that word in that class / total count of words in that class)

        feature_vectors_path = os.path.join(dir, 'feature_vectors')
        train = os.path.join(feature_vectors_path,
                             'train/train_output_vector.json')

        # Get the data from our pre-processing
        training_file = open(train)
        vectors = json.load(training_file)
        training_file.close()

        # Dictionary to keep track of the classes we can output and their count of vectors.
        # Ex. neg : 1200, pos : 1500
        classes = {}

        # Dictionary to keep track of the probability of each word given each class. This will be the output of this function.
        # Ex. P(fast|action)
        parameters = {}

        # To calculate the parameters, we'll have another dictionary that keeps track of the total count of each word per class.
        # Ex. {
        #   { neg: {
        #           word : count
        #           }
        #   }
        #   {
        #     pos: {
        #           word : count
        #           }
        #   }
        # }
        word_counts = {}

        # And finally to calculate the probability, for each class we'll also need a total count of tokens in that class.
        # - We'll also need the size of the vocabulary to implement Add One smoothing to our training model,
        # - so we'll keep track of each unique word we encounter regardless of class.
        # Use the imdb.vocab file we have
        vocab = set()
        # Add the words in the vocab file to the vocab set
        with open(os.path.join(dir, 'aclImdb/imdb.vocab'), 'r', encoding='utf8') as f:
            words = f.read().split()
            for word in words:
                vocab.add(word)
        class_total_counts = {}

        # Now let's loop through our training data and start building our model
        for v in vectors:
            # The string class the vector belongs to
            vector_class = v[0]
            # Add one to the class count
            try:
                classes[vector_class] += 1
            except KeyError:
                classes[vector_class] = 1
            # If the vector class is not in the word_counts dictionary, add it
            if vector_class not in word_counts:
                word_counts[vector_class] = {}
            # Same for the class_total_counts dictionary
            if vector_class not in class_total_counts:
                class_total_counts[vector_class] = 0
            # The dict object containing each word in the vector and its count
            vector_features = v[1]
            # For each word we're going to add the count of that word to the total count in word_counts for that label
            for word in vector_features:
                try:
                    word_counts[vector_class][word] += vector_features[word]
                except KeyError:
                    word_counts[vector_class][word] = vector_features[word]
                # We're also going to keep track of the total number of words in this class
                class_total_counts[vector_class] += vector_features[word]

        # Now to calculate probabilities and add them to the parameters dictionary
        # - Let's start with prior probabilities
        vector_total_count = len(vectors)
        for c in classes:
            key = 'P(' + c + ')'
            val = classes[c] / vector_total_count
            log_prob = math.log(val, 2)
            parameters[key] = log_prob

        vocabulary_size = len(vocab)
        # Now we'll go word by word in the vocabulary and calculate it's probability given each class
        for word in vocab:
            for c in classes:
                key = 'P(' + word + '|' + c + ')'
                val = 0
                # Add one smoothing
                if word not in word_counts[c]:
                    val = (1)/(class_total_counts[c] + vocabulary_size)
                else:
                    val = (word_counts[c][word] + 1) / \
                        (class_total_counts[c] + vocabulary_size)
                # We'll be storing the log probabilities as parameters
                log_prob = math.log(val, 2)
                parameters[key] = log_prob

        # Finally, we've trained the model and its parameters. Now to output the parameters to a file
        param_file = os.path.join(dir, 'movie-review-BOW.NB')
        with open(param_file, 'w', encoding='utf8') as output:
            output.write(json.dumps(parameters, indent=2))

        # Now lets test our model, we're doing it here so we can pass in the classes dictionary that we need in the test
        self.test_model(dir, classes)

    def test_model(self, dir, classes):

        # To test our model, we'll take the vectors and apply the following steps to each vector:
        # - For each class, log prior prob + the sum of the log probabilities of each word given that class

        # Get the data from our training
        param_path = os.path.join(dir, 'movie-review-BOW.NB')
        param_file = open(param_path)
        parameters = json.load(param_file)
        param_file.close()

        # Get the data from our pre-processing
        feature_vectors_path = os.path.join(dir, 'feature_vectors')
        test = os.path.join(feature_vectors_path,
                            'test/test_output_vector.json')
        test_file = open(test)
        vectors = json.load(test_file)
        test_file.close()

        # To find our accuracy later, we'll take the count of correct predictions / count of all predictions
        count_predictions = 0
        count_correct_predictions = 0

        # We'll be printing our results to this file as we calculate them
        output_file_path = os.path.join(dir, 'output.txt')
        output_file = open(output_file_path, 'w')

        # For each vector in the training data set
        for vector in vectors:
            # Get the correct class
            correct_class = vector[0]
            # Get the document of words
            document = vector[1]
            # We'll store the final sum of probabilities for each class here to find the largest later
            class_prob = {}
            # For each class we have
            for c in classes:
                # Start the sum at the probability of the current class we're looking at
                prior_prob = 'P(' + c + ')'
                log_prob = parameters[prior_prob]
                # For every word in the test vector's document
                for word in document:
                    # The word could happen more than once, so we'll add it to the sum as many times as it occurs
                    for i in range(document[word]):
                        # Add the word's log probability to the sum
                        parameter = 'P(' + word + '|' + c + ')'
                        log_prob += parameters[parameter]
                class_prob[c] = log_prob

            # The prediction of our model is the largest probability of all classes
            prediction = max(class_prob, key=class_prob.get)
            # If the prediction is correct, then we want to increment count_correct_predictions for our accuracy score
            if prediction == correct_class:
                count_correct_predictions += 1
            count_predictions += 1
            # And finally, we can output the result for this test vector
            text = 'Prediction: ' + prediction + ' ||  Actual: ' + correct_class
            for x in class_prob:
                text += (' || ' + x + ' : ')
                text += str(class_prob[x]) + ' '
            output_file.write(text + '\n')

        # Output summary
        accuracy_score = count_correct_predictions/count_predictions
        output_file.write('Total correct predictions: ' +
                          str(count_correct_predictions) + '\n')
        output_file.write('Total predictions: ' +
                          str(count_predictions) + '\n')
        output_file.write('Accuracy Score: ' + str(accuracy_score) + '\n')
        print('Accuracy Score: ', accuracy_score)


nb = NaiveBayes()
