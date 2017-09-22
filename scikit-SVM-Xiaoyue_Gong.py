#!/usr/bin/env python3
"""Machine Learning Hw3 - XiaoyueGong
SVM
"""
import time
import csv
import numpy as np
#from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

def scale(v):
    """returns a normalized version of v by applying (2*v/255 - 1) element-wise"""
    return 2*v/255 - 1

def vectors_from_file(fname):
    """returns a list of normalized vectors loaded from fname"""
    with open(fname, 'r') as fin:
        data = [line.split(',') for line in fin]
    norm_val = []
    for line in data:
        #avoid label at index 0
        v = scale(np.array([int(line[i]) for i in range(1, len(line))]))
        norm_val.append(v)
    return norm_val

def labels_from_file(fname):
    """returns a vector containing the label for each line from fname"""
    with open(fname, 'r') as fin:
        data = fin.readlines()
    return np.array([int(line[0]) for line in data])
"""
def cross_validate(X, Y):
	classif = OneVsRestClassifier(SVC(kernel='###'))
    classif.fit(X, Y)
	model.fit([list of training FVs], [list of labels])
	"""

def main():
    # SETUP
    training_set = 'HW3_handwriting data/mnist_train.txt'
    training_vectors = vectors_from_file(training_set)
    training_labels = labels_from_file(training_set)
    test_set = 'HW3_handwriting data/mnist_test.txt'
    test_vectors = vectors_from_file(test_set)
    test_labels = labels_from_file(test_set)

    # Straightforward (no crossval)
    model = SVC(C=1.0)
    model.fit(training_vectors,training_labels)
    predictions = model.predict(test_vectors)
    correct = 0
    for i in range(len(test_labels)):
    	if test_labels[i] == predictions[i]:
    		correct += 1
    print( 1 - ( correct / len(predictions) ) )

    total_vectors = training_vectors + test_vectors
    total_lables = np.concatenate((training_labels,test_labels), axis=0)
    X_train, X_test, y_train, y_test = train_test_split(total_vectors, total_lables, test_size=0.4, random_state=0)
    # clf = SVC(kernel='rbf', C=25,gamma=0.0007).fit(X_train, y_train)
    # clf_predictions = model.predict(X_test)
    # clf_correct = 0
    # for i in range(len(y_test)):
    #     if y_test[i] == clf_predictions[i]:
    #         clf_correct += 1
    # print( 1 - ( clf_correct / len(clf_predictions) ) )
    # #total_lables.shape()

    # scores = cross_val_score(clf, total_vectors, total_lables, cv=5)
    # print("scores =", scores)
    # ave_error_rate = 1 - sum(scores[i] for i in range(len(scores)))/5
    # print("average error rate =", ave_error_rate)

    c_val = 1
    #iterate through c_val to find a good c_val
    while c_val < 30: 
        clf = SVC(kernel='rbf', C=c_val,gamma=0.0007).fit(X_train, y_train)
        # clf_predictions = model.predict(X_test)
        # clf_correct = 0
        # for i in range(len(y_test)):
        #     if y_test[i] == clf_predictions[i]:
        #         clf_correct += 1
        # print( 1 - ( clf_correct / len(clf_predictions) ) )
        #total_lables.shape()

        scores = cross_val_score(clf, total_vectors, total_lables, cv=5)
        print("scores =", scores)
        ave_error_rate = 1 - sum(scores[i] for i in range(len(scores)))/5
        print("average error rate when c_val =", c_val, ave_error_rate)
        c_val +=0.5

    

    gam_val = 0.0007

    #iterate through gam_val for a particular good C_val
    while gam_val < 0.001: 
        clf = SVC(kernel='rbf', C=21,gamma=gam_val).fit(X_train, y_train)
        # clf_predictions = model.predict(X_test)
        # clf_correct = 0
        # for i in range(len(y_test)):
        #     if y_test[i] == clf_predictions[i]:
        #         clf_correct += 1
        # print( 1 - ( clf_correct / len(clf_predictions) ) )
        #total_lables.shape()

        scores = cross_val_score(clf, total_vectors, total_lables, cv=5)
        print("scores =", scores)
        ave_error_rate = 1 - sum(scores[i] for i in range(len(scores)))/5
        print("average error rate when gam_val =", gam_val, ave_error_rate)
        gam_val += 0.00001


    """kf = KFold(n_splits=10)
    total_lables = np.concatenate((training_labels,test_labels), axis=0)
    total_vectors = training_vectors + test_vectors
    total_set = list(zip(total_vectors, total_lables))
    print(total_set)
    # for i in range(len(total_vectors)):
        # total_set.append([total_vectors[i], total_lables[i]])
    print(len(training_vectors), len(test_vectors), len(total_vectors), len(total_set))
    kf.split(total_set)
    for train, test in kf.split(total_set):
        print(len(train), len(test))
        #model2 = SVC()
        #model2.fit(train,train[1])
        print("haha")"""


    # loop over the splits, calculate the error, average it


"""
    # TRAINING
    #container for n and average validation error
    err_data = []

"""

if __name__ == '__main__':
    try:
        start = time.time()
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print("---completed in %ss---" % str(time.time() - start))
