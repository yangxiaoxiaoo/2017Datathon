import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import svm
from sklearn import linear_model
from sklearn import tree
from sklearn import naive_bayes

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
# from sklearn.naive_bayes import GaussianNB


def pd_to_np(data):
    #x_test = data.iloc[:, [0]]
    return pd.DataFrame(data).as_matrix()

def split_train_test(X, y, ratio, is_random, seed):
    if is_random:
        X_train, X_test, y_train, y_test = train_test_split(X, 
        y, test_size = ratio, random_state = seed)
    else:
        n_samples, n_features = X.shape
        n_train = int(n_samples * ratio)
        X_train = X[:-n_train]
        X_test = X[-n_train:]
        y_train = y[:-n_train]
        y_test = y[-n_train:]
    return X_train, X_test, y_train, y_test


def pre_process(X_train, debug = 0):
	# add dummy variables
    
    # remove invalid inputs

    # mean 0, variance 1
    scaler = StandardScaler().fit(X_train)
    X_train_transformed = scaler.transform(X_train)   
    
    
    if debug:
        # test preprocessing performance, replace the learning model
        clf = make_pipeline(preprocessing.StandardScaler(), svm.SVC(C = 1))
        print(cross_val_score(clf, X_train, y_train, cv = 5))    
        clf = svm.SVC(C = 1).fit(X_train_transformed, y_train)
        X_test_transformed = scaler.transform(X_test)
        print(clf.score(X_test_transformed, y_test))
    return X_train_transformed, scaler


def select_reg_model(X, y, option, debug = 0, para = 1, para_p = 'squared_loss'):
    # 0-6 for normal; 7-10 for large outliers; 11-12 for multiple targets
    if option == 0: # ordinary least square
        model = linear_model.LinearRegression()        
    elif option == 1: # ridge
        model = linear_model.Ridge(alpha = para)
    elif option == 2: # lasso
        model = linear_model.Lasso(alpha = para)
    elif option == 3: # Elastic-net is useful when there are multiple features 
        # which are correlated with one another.
        model = linear_model.ElasticNet()
    elif option == 4: # support vector machine regression
        model = svm.SVR()
    elif option == 5: # bayesian ridge
        # same diviation on coordinates
        model = linear_model.BayesianRidge()
    elif option == 6: # Automatic Relevance Determination, 
        # different diviation on coordinates 
        model = linear_model.ARDRegression(compute_score=True)
    elif option == 7: # random sample consensus
        model = linear_model.RANSACRegressor()
    elif option == 8: # median, not useful for high dimension
        model = linear_model.TheilSenRegressor()
    elif option == 9: # huber, linear loss for large outliers
        model = linear_model.HuberRegressor()
    elif option == 10: # sgd
        model = linear_model.SGDRegressor(loss = para_p)
    elif option == 11: # matrix y, more than one target
        model = linear_model.MultiTaskLasso(alpha = para)
    elif option == 12: 
        model = linear_model.MultiTaskElasticNet(alpha = para)
    
    
    if debug == 1:
        if option == 1:
            model = linear_model.RidgeCV(alphas = para)
        elif option == 2:
            model = linear_model.LassoCV(alphas = para)
        elif option == 3:
            model = linear_model.ElasticNetCV(alphas = para)
    
    model.fit(X, y)
    return model



def select_class_model(X, y, option, debug = 0, para_p = 'l2', para = 1):
    if option == 0: # logistic regression
        model = linear_model.LogisticRegression(penalty = para_p)
    elif option == 1: # perceptron
        model = linear_model.Perceptron()
    elif option == 2: # svm
        model = svm.SVC()
        # sample_weight
        # cache_size can be set to 500 or 1000
    elif option == 3: # naive Bayes 
        model = naive_bayes.GaussianNB()
    elif option == 4:
        model = naive_bayes.MultinomialNB()
    return model

#sklearn.tree.DecisionTreeRegressor
#sklearn.tree.DecisionTreeClassifier
# sklearn.ensemble.RandomForestClassifier
# more on ensemble

def cross_val:
	# cross_validate
	# cross_val_score
