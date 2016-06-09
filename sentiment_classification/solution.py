from sklearn import svm, linear_model
from sklearn.cross_validation import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np
import pickle
import sys

#start parse_vocabulary
def parse_vocabulary(file):
    with open(file, 'rb') as f:
        vocab = pickle.load(f)
    return vocab
#end

#start read_training_set(pos_tweets, neg_tweets)
def read_training_set(pos_tweets, neg_tweets, embeddings, id_map):
    training_set, labels = [], []
    
    if len(embeddings) == 0:
        for tweet in pos_tweets:
            training_set.append(tweet.strip())
            labels.append('1')
        for tweet in neg_tweets:
            training_set.append(tweet.strip())
            labels.append('-1')
    else:
        for tweet in neg_tweets:
            features = np.array((extract_features(tweet, embeddings, id_map))).tolist()
            training_set.append(features)
            labels.append('-1')
        for tweet in pos_tweets:
            features = np.array((extract_features(tweet, embeddings, id_map))).tolist()
            training_set.append(features)
            labels.append('1')
            
    return training_set, labels
#end

#start get_word_vector
def get_word_vector(word, embeddings, id_map):
    # Returns word vector from generated embeddings
    word_vector = np.array([])
    # If word exists in word_to_id dictionary, then grab the corresponding embeddings vector
    if word in id_map:
        embd_index = id_map[word]
        word_vector = embeddings[embd_index,:]
        
    return word_vector
#end

#start extract_features
def extract_features(tweet, embeddings, id_map):
    tweet_words = set(tweet)
    features = np.array(embeddings.shape[0])
    word_count = 0
    for word in tweet_words:
        # get word from embeddings
        feature = get_word_vector(word, embeddings, id_map)
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

#start predict_classification
def predict_classification(classifier, vectorizer, embeddings, id_map, tweets):
    # Classify tweets
    if vectorizer != 0: # If tf_idf was used
        return classifier.predict(vectorizer.transform(tweets))
    else:
        feature_list = []
        for tweet in tweets:
            feature_list.append(extract_features(tweet, embeddings, id_map))
        
        return classifier.predict(feature_list)
#end

#start cross_validate
def cross_validate(classifier, features_test, labels_test):
    # Cross Validation of Results
    return cross_val_score(classifier, features_test, labels_test, cv=5)
#end

#start classify_dataset
def classify_dataset(path_to_dataset, path_to_export, classifier, embeddings, vectorizer, id_map):
    csvfile = open(path_to_export, 'w')
    classified_tweets = []
    
    with open(path_to_dataset, 'r') as dataset:
        tweets = []
        for line in dataset:
            id, tweet = line.split(',', 1)
            tweets.append(tweet)
    
    if vectorizer != 0:
        classified_tweets = predict_classification(classifier, vectorizer, 0, id_map, tweets)
    else:
        classified_tweets = predict_classification(classifier, 0, embeddings, id_map, tweets)
    
    id = 1
    csvfile.write('Id,Prediction\n')
    for prediction in classified_tweets:
        csvfile.write(str(id) + ',' + prediction + '\n')
        id += 1
        
    csvfile.close()
#end

def main():
    tf_idf = 0 # true/false if tf-idf is used
    try:
        if (sys.argv[1] == '-tfidf'):
            tf_idf = 1
            print('Tf-Idf enabled')
    except IndexError:
        print('Tf-Idf disabled')
    
    id_map = parse_vocabulary('assets/vocab.pkl')
    embeddings = np.load('assets/embeddings.npy')

    # Read the tweets and prepare them for classifier
    pos_tweets = open('assets/train_pos_1perc.txt', mode='r', encoding="utf8")
    neg_tweets = open('assets/train_neg_1perc.txt', mode='r', encoding="utf8")
    training_set, labels = \
        read_training_set(pos_tweets, neg_tweets, [], id_map) if tf_idf == 1 else read_training_set(pos_tweets, neg_tweets, embeddings, id_map) 
    
    vectorizer = 0
    if tf_idf: # If Tf-Idf is used, vectorize the data we have
        vectorizer = TfidfVectorizer()
        training_set = vectorizer.fit_transform(training_set)
        
    ###
    # 1. Train linear SVM classifier & print the results
    ###
    #classifier = svm.LinearSVC()
    #classifier.fit(training_set, training_set_labels)
    
    #scores_svm = cross_validate(classifier, training_set, labels)
    #print("Accuracy of SVM %s" % scores_svm)
    #print("Accuracy: %0.2f (+/- %0.2f)" % (scores_svm.mean(), scores_svm.std() * 2))

    ###
    # 2. Traing logistic regression classifier  & print the results
    ###
    classifier = linear_model.LogisticRegression()
    classifier.fit(training_set, labels)
    
    scores_logReg = cross_validate(classifier, training_set, labels)
    print("Accuracy of log regression: %s" % scores_logReg)
    print("Acuracy: %0.2f (+/- %0.2f)" % (scores_logReg.mean(), scores_logReg.std() * 2))

    # Classify the dataset given to CSV file
    classify_dataset('assets/test_data.txt', 'submissions/classified_test_data.csv', classifier, embeddings, vectorizer, id_map)

if __name__ == '__main__':
    main()
