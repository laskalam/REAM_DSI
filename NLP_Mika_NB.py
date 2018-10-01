import sys
import numpy as np
import pandas as pd
from glob import glob
from sklearn import metrics, model_selection, naive_bayes, preprocessing
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as mpl

classes = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

train_data = 'train.csv'
test_data = 'test.csv'

#Get reviews and labels
train_data = pd.read_csv(train_data).fillna(' ')
test_data = pd.read_csv(test_data).fillna(' ')
#Combine all text to fit
text_train = train_data['comment_text']
text_test = test_data['comment_text']
text_comb = pd.concat([text_train, text_test])
labels = train_data[classes].values



#Split into training and validation
#train_comment, valid_comment, train_label, valid_label = \
#             model_selection.train_test_split(text, labels,
#             random_state=1234567, test_size=0.2)

#train_id = train_comment[:,0]
#train_comment = train_comment[:,1]
#valid_id = valid_comment[:,0]
#valid_comment = valid_comment[:,1]

##Convert the labels to 0, 1
#convert = preprocessing.LabelEncoder()
#train_label = convert.fit_transform(train_label)
#valid_label = convert.fit_transform(valid_label)

# create a count vectorizer object 
#name = 'CountVec'
#count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
#count_vect.fit(text_comb)

# transform the training and validation data using count vectorizer object
#xtrain_ =  count_vect.transform(text_train)
#xtest_ =  count_vect.transform(text_test)
#xvalid_count =  count_vect.transform(valid_comment)

## word level tf-idf
name = 'Tfidf'
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(text_comb)
xtrain_ =  tfidf_vect.transform(text_train)
xtest_ =  tfidf_vect.transform(text_test)
#xvalid_ =  tfidf_vect.transform(valid_comment)
#
## ngram level tf-idf 
#tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
#tfidf_vect_ngram.fit(text[:,1])
#xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_comment)
#xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_comment)
#
## characters level tf-idf
#tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
#tfidf_vect_ngram_chars.fit(text[:,1])
#xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_comment) 
#xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_comment) 

def train_model(Classifier, features_train, target, classs, features_test, cross_validation=2):
    classifier = clone(Classifier)
    score = np.mean(cross_val_score(classifier, features_train, target, cv=cross_validation, scoring='roc_auc'))
    sys.stdout.write('%s score:%6.4f\n'%(classs, score))
    classifier.fit(features_train, target)
    #return classifier.predict(features_test)
    return classifier.predict_proba(features_test)[:,1]

Outputs = 'NaiveBayes_%s.csv'%name
output = pd.DataFrame.from_dict({'id': test_data['id']})
for c in classes:
    output[c] = train_model(naive_bayes.MultinomialNB(), xtrain_, train_data[c], c, xtest_)

output.to_csv(Outputs, index=False)

#Outputs = 'NaiveBayes.txt'
#file_ = open(Outputs, 'w')
#Naive Bayes
# Naive Bayes on Count Vectors
#accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_count, train_label, xvalid_count, valid_label, nlabel=len(classes))
#print "NB, Count Vectors: %6.4f", accuracy
#file_.write("NB, Count Vectors: %6.4f\n"%accuracy)

## Naive Bayes on Word Level TF IDF Vectors
#accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_label, xvalid_tfidf, valid_label)
#file_.write("NB, WordLevel TF-IDF: %6.4f\n"%accuracy)
#
## Naive Bayes on Ngram Level TF IDF Vectors
#accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, train_label, xvalid_tfidf_ngram, valid_label)
#file_.write("NB, N-Gram Vectors: %6.4f\n"%accuracy)
#
## Naive Bayes on Character Level TF IDF Vectors
#accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram_chars, train_label, xvalid_tfidf_ngram_chars, valid_label)
#file_.write("NB, CharLevel Vectors: %6.4f\n"%accuracy)
#file_.close()
sys.stdout.write('%s saved\n'%Outputs)
