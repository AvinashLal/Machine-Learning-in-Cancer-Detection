import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import time

def shuffleLists(list1, list2):
      shuffle = np.random.permutation(list1.shape[0])
      return list1[shuffle], list2[shuffle]

def labelToNum(label, cancerType, allCancers):
      if label == 'Solid Tissue Normal':
            return 0
      if label == 'Primary Tumor':
            return 1
      if label == 'Recurrent Tumor':
            return 1
      if label == 'Metastatic':
            return 1
      if label == 'Primary Blood Derived Cancer - Bone Marrow':
            return 1
      if label == 'Primary Blood Derived Cancer - Peripheral Blood':
            return 1
      if label == 'Recurrent Blood Derived Cancer - Peripheral Blood':
            return 1
      if label == 'Additional - New Primary':
            return 1
      if label == 'Additional Metastatic':
            return 1
      raise ValueError('Could not find label: ' + label)

def labelToNumV2(label, cancerType, allCancers):
      if label == 'Solid Tissue Normal':
            return 0
      if allCancers.index(cancerType) >= 20:
            return allCancers.index(cancerType)
      return 1 + allCancers.index(cancerType)

def printStats(correct_ans, prediction):
      print("Accuracy: " + str(metrics.accuracy_score(correct_ans, prediction)))
      print("Recall: " + str(metrics.recall_score(correct_ans, prediction, average='weighted')))
      print("Precision: " + str(metrics.precision_score(correct_ans, prediction, average='weighted')))

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.viridis):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = 'Confusion Matrix (Normalized)'

    plt.figure(figsize = (10,10))
    sns.set_style("whitegrid", {'axes.grid' : False})
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title,fontsize=25)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    plt.tight_layout()
    plt.ylabel('True label',fontsize=15)
    plt.xlabel('Predicted label',fontsize=15)
    plt.show()

print('Reading Data...')
data = pd.read_csv('rnaseq_data.csv',header=0,index_col=0).fillna(0)
mapping = pd.read_csv('rnaseq_mapping.csv',header=0,index_col=0)

print('Processing Data...')
data_cols = data.columns.tolist()
allcancers = list(set(mapping['cases.0.project.project_id'].tolist()))

features = data.T.as_matrix()
labels = np.asarray([labelToNumV2(mapping.loc[x][0], mapping.loc[x][1], allcancers) for x in data_cols])
features, labels = shuffleLists(features, labels)

numTest = 1000

trainstart = time.time()
print('Training Model...')
if 'clf' in locals():
      del clf
#clf = GaussianNB()
clf = SVC(gamma=0.001, C=100., kernel='linear')
clf.fit(features[:-numTest], labels[:-numTest])
trainend = time.time()

predictstart = time.time()
print('Doing prediction...')
prediction = clf.predict(features[-numTest:])
correct_ans = labels[-numTest:]
predictend = time.time()

print('Done!\n')
print("TIme to Train: " + str(trainend - trainstart) + " seconds")
print("TIme to Predict: " + str(predictend - predictstart) + " seconds")
print('\n')
printStats(correct_ans, prediction)

labellist = list(allcancers)
labellist.insert(0, 'Normal')
activelabels = np.array(labellist)[list(set(correct_ans).union(set(prediction)))]
confmat = metrics.confusion_matrix(correct_ans, prediction)
normalized_confmat = confmat.astype('float') / confmat.sum(axis=1)[:, np.newaxis]

print(metrics.classification_report(correct_ans, prediction,target_names=activelabels))
print('Target    \tAccuracy')
for x in range(len(normalized_confmat)):
      print(activelabels[x] + '  \t' + str(normalized_confmat[x][x]))

plot_confusion_matrix(confmat, activelabels, normalize = False)
plot_confusion_matrix(confmat, activelabels, normalize = True)
