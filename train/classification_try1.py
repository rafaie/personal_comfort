
import sys
import os
import datetime
import csv
import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neighbors import NearestCentroid

import warnings
warnings.filterwarnings(action='ignore')


def do_cross_validation(ML_NAME, clf, X_train, Y_train, tain_type, csv_out,
                        f_name, param1, n_jobs=4, cv=4):
    score = np.mean(cross_val_score(clf, X_train, Y_train, n_jobs=n_jobs,
                                    cv=cv))
    r = [ML_NAME, tain_type, param1, f_name, score]
    csv_out.writerow(r)
    print(r)

    return score


def do_classification(clm, data_fname, clm_type):
    d_name = "output"
    if os.path.isdir(d_name) is False:
        os.mkdir(d_name)

    fname = os.path.basename(data_fname).replace('.csv', '')
    fn = 'result_' + fname + "_type" + str(clm_type) + "_" +\
        datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + '.csv'
    csv_out_fnamee = os.path.join(d_name, fn)
    fi = open(csv_out_fnamee, 'w')
    csv_out = csv.writer(fi, delimiter=',')

    # Create dataframe for training
    base_df = pd.read_csv(data_fname)
    df = base_df[clm]

    df = df[df['heartRate'] > 40]
    df = df[df['skinTemperature'] > 10]
    df = df[df['met'] > 0.4]

    X_train = df[clm[:-2]]
    Y_train = [df[clm[-2]], df[clm[-1]]]

    # Model: Decision Tree
    ML_NAME = 'Decision Tree'
    depth_list = np.concatenate((np.arange(1, 10), np.arange(10, 20, 2),
                                 np.arange(20, 50, 5), np.arange(50, 100, 10),
                                 np.arange(150, 1000, 50)))
    for t in [0, 1]:
        for depth in depth_list:
            clf = DecisionTreeClassifier(class_weight=None,
                                         criterion='entropy',
                                         max_depth=depth,
                                         max_features=None,
                                         max_leaf_nodes=None,
                                         min_impurity_decrease=0.0,
                                         min_impurity_split=None,
                                         min_samples_leaf=1,
                                         min_samples_split=2,
                                         min_weight_fraction_leaf=0.0,
                                         presort=False,
                                         random_state=None,
                                         splitter='best')
            do_cross_validation(ML_NAME, clf, X_train, Y_train[t], t, csv_out,
                                fname, depth)

    # Model: Extra Tree Classifier
    ML_NAME = 'Extremely randomized tree classifier'
    for t in [0, 1]:
        clf = ExtraTreeClassifier()
        do_cross_validation(ML_NAME, clf, X_train, Y_train[t], t, csv_out,
                            fname, 0)

    # Model: Gaussian
    ML_NAME = 'Gaussian Naive Bayes'

    for t in [0, 1]:
        clf = GaussianNB(priors=None)
        do_cross_validation(ML_NAME, clf, X_train, Y_train[t], t, csv_out,
                            fname, 0)

    # Model: Multivariate Bernoulli Model
    ML_NAME = 'Multivariate Bernoulli Model'
    alphas = np.concatenate((np.arange(0.1, 1, 0.2), np.arange(1, 10),
                             np.arange(10, 20, 2), np.arange(20, 50, 5),
                             np.arange(50, 150, 10)))

    for t in [0, 1]:
        for a in alphas:
            clf = BernoulliNB(alpha=a)
            do_cross_validation(ML_NAME, clf, X_train, Y_train[t], t, csv_out,
                                fname, a)

    # Model: AdaBoost Classifier
    ML_NAME = 'AdaBoost classifier'
    noestimator = np.arange(5, 1000, 20)

    for t in [0, 1]:
        for n in noestimator:
            clf = AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
                                     learning_rate=0.1, n_estimators=n,
                                     random_state=None)
            do_cross_validation(ML_NAME, clf, X_train, Y_train[t], t, csv_out,
                                fname, n)

    # Model: Gradient Boosting Classifier
    ML_NAME = 'Gradient Boosting Classifier'
    noestimator = np.arange(5, 1000, 20)

    for t in [0, 1]:
        for n in noestimator:
            clf = GradientBoostingClassifier(criterion='friedman_mse',
                                             init=None,
                                             learning_rate=0.1,
                                             loss='deviance',
                                             max_depth=4,
                                             max_features=None,
                                             max_leaf_nodes=None,
                                             min_impurity_decrease=0.0,
                                             min_impurity_split=None,
                                             min_samples_leaf=1,
                                             min_samples_split=2,
                                             min_weight_fraction_leaf=0.0,
                                             n_estimators=n,
                                             presort='auto',
                                             random_state=None,
                                             subsample=1.0,
                                             verbose=0,
                                             warm_start=False)
            do_cross_validation(ML_NAME, clf, X_train, Y_train[t], t, csv_out,
                                fname, n)

    # Model: Random Forest Classifier
    ML_NAME = 'Random Forest Classifier'
    noestimator = np.concatenate((np.arange(1, 10), np.arange(10, 20, 2),
                                  np.arange(20, 50, 5),
                                  np.arange(50, 150, 10),
                                  np.arange(150, 1000, 50)))

    for t in [0, 1]:
        for n in noestimator:
            clf = AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
                                     learning_rate=0.1, n_estimators=n,
                                     random_state=None)
            do_cross_validation(ML_NAME, clf, X_train, Y_train[t], t, csv_out,
                                fname, n)

    # Model: Support Vector Machines - RBF
    ML_NAME = 'Support Vector Machines - RBF'
    c_values = np.concatenate((np.arange(0.1, 1, 0.2), np.arange(1, 10),
                               np.arange(10, 20, 2), np.arange(20, 50, 5),
                               np.arange(50, 150, 10)))

    for t in [0, 1]:
        for c in c_values:
            clf = SVC(C=c, kernel='rbf')
            do_cross_validation(ML_NAME, clf, X_train, Y_train[t], t, csv_out,
                                fname, c)

    # Model: Support Vector Machines - poly
    ML_NAME = 'Support Vector Machines - poly'
    c_values = np.concatenate((np.arange(0.1, 1, 0.2), np.arange(1, 10),
                               np.arange(10, 20, 2), np.arange(20, 50, 5),
                               np.arange(50, 150, 10)))

    for t in [0, 1]:
        for c in c_values:
            clf = SVC(C=c, kernel='poly')
            do_cross_validation(ML_NAME, clf, X_train, Y_train[t], t, csv_out,
                                fname, c)

    # Model: Support Vector Machines - Sigmoid
    ML_NAME = 'Support Vector Machines - Sigmoid'
    c_values = np.concatenate((np.arange(0.1, 1, 0.2), np.arange(1, 10),
                               np.arange(10, 20, 2), np.arange(20, 50, 5),
                               np.arange(50, 150, 10)))

    for t in [0, 1]:
        for c in c_values:
            clf = SVC(C=c, kernel='sigmoid')
            do_cross_validation(ML_NAME, clf, X_train, Y_train[t], t, csv_out,
                                fname, c)

    # Model: Support Vector Machines - Linear
    ML_NAME = 'Support Vector Machines - Linear'
    c_values = np.concatenate((np.arange(0.1, 1, 0.2), np.arange(1, 10),
                               np.arange(10, 20, 2), np.arange(20, 50, 5),
                               np.arange(50, 150, 10)))

    for t in [0, 1]:
        for c in c_values:
            clf = SVC(C=c, kernel='linear')
            do_cross_validation(ML_NAME, clf, X_train, Y_train[t], t, csv_out,
                                fname, c)

    # Model: KNeighborsClassifier
    ML_NAME = 'k-nearest neighbors Classifier'
    n_neighbors = np.concatenate((np.arange(1, 10),
                                  np.arange(10, 20, 2), np.arange(20, 50, 5),
                                  np.arange(50, 150, 10)))

    for t in [0, 1]:
        for n in n_neighbors:
            if n < len(X_train)/2:
                try:
                    clf = KNeighborsClassifier(n_neighbors=n)
                    do_cross_validation(ML_NAME, clf, X_train, Y_train[t], t,
                                        csv_out, fname, n)
                except:
                    pass

    # Model: RadiusNeighborsClassifier
    ML_NAME = 'Nearest Centroid Classifier'

    for t in [0, 1]:
        clf = NearestCentroid()
        do_cross_validation(ML_NAME, clf, X_train, Y_train[t], t,
                            csv_out, fname, 0)


clmns = 5 * ['']
clmns[0] = ['roomTempreture', 'roomHumidity', 'heartRate', 'skinTemperature',
            'clothingScore', 'met', 'resistance', 'vote2', 'vote3']

clmns[1] = ['roomTempreture', 'roomHumidity', 'heartRate', 'skinTemperature',
            'clothingScore', 'met', 'vote2', 'vote3']

clmns[2] = ['roomTempreture', 'roomHumidity', 'heartRate', 'skinTemperature',
            'clothingScore', 'met', 'resistance', 'met_15min',
            'roomTempreture_15min', 'roomHumidity_15min', 'resistance_15min',
            'heartRate_15min', 'skinTemperature_15min', 'met_30min',
            'roomTempreture_30min', 'roomHumidity_30min', 'resistance_30min',
            'heartRate_30min', 'skinTemperature_30min', 'vote2', 'vote3']

clmns[3] = ['roomTempreture', 'roomHumidity', 'heartRate', 'skinTemperature',
            'clothingScore', 'met', 'met_15min',
            'roomTempreture_15min', 'roomHumidity_15min',
            'heartRate_15min', 'skinTemperature_15min', 'met_30min',
            'roomTempreture_30min', 'roomHumidity_30min',
            'heartRate_30min', 'skinTemperature_30min', 'vote2', 'vote3']


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Please run program with this format:\n" +
              "python classification_try1.py [clm_type:int] [data file path]" +
              "\n")
        sys.exit(1)

    clm_type = int(sys.argv[1])
    clm = clmns[clm_type]
    print("Start Processing", sys.argv[2], " With clmn = ", clm)
    do_classification(clm, data_fname=sys.argv[2], clm_type=clm_type)
