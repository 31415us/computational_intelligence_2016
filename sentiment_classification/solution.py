from sklearn import svm, linear_model
from sklearn.cross_validation import cross_val_score

# Tf-Idf
from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np
import pickle

# Initialise globals
tf_idf = 1 # true/false if tf-idf is used
word_to_id = {}

#start parse_vocabulary
def parse_vocabulary(file):
    with open(file, 'rb') as f:
        vocab = pickle.load(f)

    count = 0
    for w in vocab:
        word_to_id[w] = count
        count += 1
#end

#start read_training_set(pos_tweets, neg_tweets)
def read_training_set(pos_tweets, neg_tweets, embeddings):
    training_set, labels = [], []
    if tf_idf:
        for tweet in pos_tweets:
            training_set.append(tweet.strip())
            labels.append('positive')
        for tweet in neg_tweets:
            training_set.append(tweet.strip())
            labels.append('negative')
    else:
        for tweet in neg_tweets:
            features = np.array((extract_features(tweet, embeddings))).tolist()
            training_set.append(features)
            labels.append('negative')
        for tweet in pos_tweets:
            features = np.array((extract_features(tweet, embeddings))).tolist()
            training_set.append(features)
            labels.append('positive')

    return training_set, labels

#start get_word_vector
def get_word_vector(word, embeddings):
    # Returns word vector from generated embeddings
    word_vector = np.array([])

    # If word exists in word_to_id dictionary, then grab the corresponding embeddings vector
    if word in word_to_id:
        embd_index = word_to_id[word]
        word_vector = embeddings[embd_index,:]

    return word_vector
#end

#start extract_features
def extract_features(tweet, embeddings):
    tweet_words = set(tweet)

    features = np.array(embeddings.shape[0])
    word_count = 0
    for word in tweet_words:
        # get word from embeddings
        feature = get_word_vector(word, embeddings)

        if feature.size is 0: # If feature is empty, skip iteration
            continue

        # sum with others
        features = np.add(features, feature)
        # increase the count
        word_count += 1

    # divide by the word_count
    np.divide(features, word_count)

    return features
#end

#start print_classified
def print_classified(classifier, embeddings):
    # Extract all tweets to classify
    testTweet = "<user> yep gonna watch it now"
    testTweet1 = "i'm glad this day is ending"
    testTweet2 = "why is she so perfect <url>"
    testTweet3 = "why do i get treated like this"

    print(classifier.predict(extract_features(testTweet, embeddings)))
    print(classifier.predict(extract_features(testTweet1, embeddings)))
    print(classifier.predict(extract_features(testTweet2, embeddings)))
    print(classifier.predict(extract_features(testTweet3, embeddings)))
#end

#start print_classified_with_tfidf
def print_classified_with_tfidf(classifier, vectorizer):
    # Extract all tweets to classify
    testTweet = "<user> yep gonna watch it now"
    testTweet1 = "i'm glad this day is ending"
    testTweet2 = "why is she so perfect <url>"
    testTweet3 = "why do i get treated like this"

    print(classifier.predict(vectorizer.transform([testTweet])))
    print(classifier.predict(vectorizer.transform([testTweet1])))
    print(classifier.predict(vectorizer.transform([testTweet2])))
    print(classifier.predict(vectorizer.transform([testTweet3])))
#end

#start predict_classification
def predict_classification(classifier, vectorizer, embeddings, tweets):
    # Classify tweets
    if vectorizer != 0: # If tf_idf was used
        print(tweets[0])
        return classifier.predict(vectorizer.transform(tweets))
    else:
        feature_list = []
        for tweet in tweets:
            feature_list.append(extract_features(tweet, embeddings))
        
        return classifier.predict(feature_list)
#end

#start cross_validate
def cross_validate(classifier, features_test, labels_test):
    # Cross Validation of Results
    return cross_val_score(classifier, features_test, labels_test, cv=5)
#end

#start classify_dataset
def classify_dataset(path_to_dataset, path_to_export, classifier, embeddings, vectorizer, tfidf):
    csvfile = open(path_to_export, 'w')
    classified_tweets = []
    
    with open(path_to_dataset, 'r') as dataset:
        tweets = []
        for line in dataset:
            id, tweet = line.split(',', 1)
            tweets.append(tweet)
    
    if vectorizer != 0:
        classified_tweets = predict_classification(classifier, vectorizer, 0, tweets)
    else:
        classified_tweets = predict_classification(classifier, 0, embeddings, tweets)
    
    id = 1
    for prediction in classified_tweets:
        csvfile.write(str(id) + ',' + prediction + '\n')
        id += 1
        
    csvfile.close()
#end

def main():
    parse_vocabulary('vocab.pkl')
    embeddings = np.load('embeddings.npy')

    # Read the tweets and prepare them for classifier
    pos_tweets = open('train_pos_cut.txt', 'r')
    neg_tweets = open('train_neg_cut.txt', 'r')
    training_set, labels = read_training_set(pos_tweets, neg_tweets, embeddings)

    if tf_idf: # If Tf-Idf is used, vectorize the data we have
        vectorizer = TfidfVectorizer()
        training_set = vectorizer.fit_transform(training_set)

    # Split dataset for cross_validation
    """
    training_set, test_set, training_set_labels, test_set_labels = cross_validation.train_test_split(
        training_set, labels, test_size=0.4, random_state=0)
    """

    ###
    # Train SVM classifier & print the results
    ###
    classifier = svm.SVC()
    classifier.fit(training_set, labels)
    #print_classified_with_tfidf(classifier, vectorizer) if tf_idf else print_classified(classifier, embeddings)
    #scores_svm = cross_validate(classifier, training_set, labels)
    #print("Accuracy of SVM %s" % scores_svm)
    #print("Accuracy: %0.2f (+/- %0.2f)" % (scores_svm.mean(), scores_svm.std() * 2))

    ###
    # Traing logistic regression classifier  & print the results
    ###

    classifier = linear_model.LogisticRegression()
    classifier.fit(training_set, labels)
    #print_classified_with_tfidf(classifier, vectorizer) if tf_idf else print_classified(classifier, embeddings)
    #scores_logReg = cross_validate(classifier, training_set, labels)
    #print("Accuracy of log regression: %s" % scores_logReg)
    #print("Acuracy: %0.2f (+/- %0.2f)" % (scores_logReg.mean(), scores_logReg.std() * 2))

    classify_dataset('test_data.txt', 'classified_test_data.csv', classifier, embeddings, vectorizer, tf_idf)

if __name__ == '__main__':
    main()
