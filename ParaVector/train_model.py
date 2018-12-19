# -*- coding: utf-8 -*-
from sklearn import datasets

#load origin dataset from project
data= datasets.load_files('data',encoding='utf-8',load_content=True,shuffle=True,decode_error='ignore')
neg=data.data[0].split('\n')
pos=data.data[1].split('\n')
neg_len=len(neg)
pos_len=len(pos)
train_data=neg+pos
train_label=[0]*neg_len+[1]*pos_len
#get stop_word list from file
f=open('./source/stopwords.txt','r')
stpw_list=f.read().splitlines()

#extracting feature from text
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(stop_words=stpw_list)
X_train=count_vect.fit_transform(train_data)



#tf and idf 
from sklearn.feature_extraction.text import TfidfTransformer
#X_train_neg=TfidfTransformer(use_idf=False).fit(X_train_counts_neg)
#X_train_neg_tf=X_train_neg.transform(X_train_counts_neg)
#X_train_pos=TfidfTransformer(use_idf=False).fit(X_train_counts_pos)
#X_train_pos_tf=X_train_pos.transform(X_train_counts_pos)

#more simple to get tfidf
tfidf_transformer=TfidfTransformer()
X_train_tfidf=tfidf_transformer.fit_transform(X_train)

#divide the test_set and train_set
import sklearn.model_selection as sm
train,test,train_lable,test_lable=sm.train_test_split(X_train_tfidf,train_label,test_size=0.4)


#training a classifier
import sklearn.linear_model as lm
model=lm.LogisticRegression()
model.fit(train,train_lable)
print(model.score(test,test_lable))
#save the model
from sklearn.externals import joblib
joblib.dump(model,'./model/sm.model')

#use SVM to train model
from sklearn import svm
clf=svm.SVC(kernel='linear')
clf.fit(train,train_lable)
print(clf.score(test,test_lable))
joblib.dump(clf,'./model/clf.model')