import sys
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from scipy.sparse import hstack
nval = 0.25
class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

train = pd.read_csv('../Toxic/train.csv').fillna(' ')
test = pd.read_csv('../Toxic/test.csv').fillna(' ')

train = train.sample(frac=1, random_state=9973).reset_index(drop=True)
valid = int(nval* train.shape[0])
validation = train[:valid]
train = train[valid:]

train_text = train['comment_text']
val_text = validation['comment_text']
test_text = test['comment_text']
all_text = pd.concat([train_text, val_text, test_text])

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 1),
    max_features=10000)
word_vectorizer.fit(all_text)
train_word_features = word_vectorizer.transform(train_text)
val_word_features = word_vectorizer.transform(val_text)
test_word_features = word_vectorizer.transform(test_text)

char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    stop_words='english',
    ngram_range=(2, 6),
    max_features=50000)
char_vectorizer.fit(all_text)
train_char_features = char_vectorizer.transform(train_text)
val_char_features = char_vectorizer.transform(val_text)
test_char_features = char_vectorizer.transform(test_text)

train_features = hstack([train_char_features, train_word_features])
val_features = hstack([val_char_features, val_word_features])
test_features = hstack([test_char_features, test_word_features])

scores = []
submission = pd.DataFrame.from_dict({'id': test['id']})
#validation_ = pd.DataFrame.from_dict({'id': validation['id']})
for class_name in class_names:
    train_target = train[class_name]
    val_target = validation[class_name]
    classifier = LogisticRegression(C=0.1, solver='sag')

    cv_score = np.mean(cross_val_score(classifier, train_features, train_target, cv=3, scoring='roc_auc'))
    scores.append(cv_score)
    print 'CV score for class {} is {}'.format(class_name, cv_score)

    classifier.fit(train_features, train_target)
    submission[class_name] = classifier.predict_proba(test_features)[:, 1]
    #validation_[class_name] = classifier.predict_proba(val_features)[:,1]
    #validation_[class_name+'_orig'] = validation[class_name]
    #roc_auc = roc_auc_score(validation[class_name], validation_[class_name])
    #print 'Validation score for {} is {}'.format(class_name, roc_auc)    

print 'Total CV score is {}'.format(np.mean(scores))

#Outputs = 'Validation.csv'
Outputs = 'submissionLR.csv'
submission.to_csv(Outputs, index=False)

#validation_.to_csv(Outputs, index=False)
