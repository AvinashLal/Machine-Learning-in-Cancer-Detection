import numpy as np
import pandas as pd
from rnaparser import features,labels as rna_features,rna_labels
from methparser import features,labels as meth_features,meth_labels
from sklearn.naive_bayes import GaussianNB

def Gaussianbayes(features,labels):
    #prediction method
    clf = GaussianNB()
    clf.fit(features[:-100], labels[:-100])

    prediction = clf.predict(features[-100:])
    correct_ans = labels[-100:]
 
    #accuracy
    def percentCorrect(prediction, correct_ans):
          numcorrect = 0
          for i in range(len(prediction)):
                if prediction[i] == correct_ans[i]:
                      numcorrect += 1
          return (numcorrect * 1.0)/len(prediction)
  
    print("naive bayes is %f",percentCorrect(prediction,correct_ans))

Gaussianbayes(rna_features,rna_labels)
Gaussianbayes(meth_features,meth_labels)