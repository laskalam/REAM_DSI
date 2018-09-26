import sys
import pandas as pd
from glob import glob
from sklearn import metrics, model_selection, naive_bayes, preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import matplotlib.pyplot as mpl

reviews = '../reviews.txt'
labels = '../labels.txt'

#Get reviews and labels
reviews = pd.read_csv(reviews, header=None)
labels = pd.read_csv(labels, header=None)

#Combine reviews and labels
Train = pd.DataFrame()
Train['reviews'] = reviews.values[:,0]
Train['labels'] = labels.values[:,0]



#Split into training and validation
train_reviews, valid_reviews, train_label, valid_label = \
             model_selection.train_test_split(Train['reviews'], Train['labels'],
             random_state=1234567, test_size=0.2)

#Convert the labels to 0, 1
convert = preprocessing.LabelEncoder()
train_label = convert.fit_transform(train_label)
valid_label = convert.fit_transform(valid_label)

# create a count vectorizer object 
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(Train['reviews'])

# transform the training and validation data using count vectorizer object
xtrain_count =  count_vect.transform(train_reviews)
xvalid_count =  count_vect.transform(valid_reviews)

# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(Train['reviews'])
xtrain_tfidf =  tfidf_vect.transform(train_reviews)
xvalid_tfidf =  tfidf_vect.transform(valid_reviews)

# ngram level tf-idf 
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram.fit(Train['reviews'])
xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_reviews)
xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_reviews)

# characters level tf-idf
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram_chars.fit(Train['reviews'])
xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_reviews) 
xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_reviews) 



def train_model(classifier, reviews_train, train_label, reviews_valid, valid_label):
    # fit the training dataset on the classifier
    classifier.fit(reviews_train, train_label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(reviews_valid)
    
    
    return metrics.accuracy_score(predictions, valid_label)


Outputs = 'NaiveBayes.txt'
file_ = open(Outputs, 'w')
#Naive Bayes
# Naive Bayes on Count Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_count, train_label, xvalid_count, valid_label)
file_.write("NB, Count Vectors: %6.4f\n"%accuracy)

# Naive Bayes on Word Level TF IDF Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_label, xvalid_tfidf, valid_label)
file_.write("NB, WordLevel TF-IDF: %6.4f\n"%accuracy)

# Naive Bayes on Ngram Level TF IDF Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, train_label, xvalid_tfidf_ngram, valid_label)
file_.write("NB, N-Gram Vectors: %6.4f\n"%accuracy)

# Naive Bayes on Character Level TF IDF Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram_chars, train_label, xvalid_tfidf_ngram_chars, valid_label)
file_.write("NB, CharLevel Vectors: %6.4f\n"%accuracy)
file_.close()
sys.stdout.write('%s saved\n'%Outputs)
