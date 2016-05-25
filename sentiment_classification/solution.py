from sklearn import svm
from sklearn import cross_validation
import numpy as np
import pickle

# initialise globals
featureList = []
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

<<<<<<< HEAD
#start cross_validate
def cross_validate(classifier, features_test, labels_test):
    # Cross Validation of Results
    return classifier.score(features_test, labels_test)
#end

=======
>>>>>>> d61ff6ebaf71dc52f8932964d383436a4d73db8a
def main():
    global featureList, embeddings
    
    word_to_id = parse_vocabulary('vocab.pkl')
    
    embeddings = np.load('embeddings.npy')
    
    #Read the tweets and process them
    tweets = []
    pos_tweets = open('train_pos_cut.txt', 'rb')
    neg_tweets = open('train_neg_cut.txt', 'rb')
    
    # Preparing feature set for classifier
    training_set = []
    labels = []
    
    for row in neg_tweets:
        sentiment = 'negative'
        tweet = row.decode('utf-8')
        features = np.array((extract_features(tweet, embeddings))).tolist()
        training_set.append(features)
        labels.append(sentiment)
    for row in pos_tweets:
        sentiment = 'positive'
        tweet = row.decode('utf-8')
        features = np.array((extract_features(tweet, embeddings))).tolist()
        training_set.append(features)
        labels.append(sentiment)
    #end loops
    
    # Split dataset for cross_validation
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        training_set, labels, test_size=0.4, random_state=0)
    
    # Train classifier
    classifier = svm.SVC()
    classifier.fit(X_train, y_train)
    
    print_classified(classifier, embeddings)
    print("Accuracy of classifier: %s" % cross_validate(classifier, X_test, y_test))
    
    
if __name__ == '__main__':
    main()