# importing libraries
import pandas as pd
import matplotlib.pyplot as plt

# import seaborn as sns
import logging
import time
import datetime
import os
 
# importing libraries for GA feature selection
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn_genetic import GAFeatureSelectionCV
from sklearn_genetic.plots import plot_fitness_evolution
from sklearn_genetic.callbacks import ProgressBar
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split


class FeatureSelectionGA:
    '''
    A class to implement feature selection using Genetic Algorithm .

    Args:
        file_path (str): path to csv file 
        target (str): target variable for classification
        drop_features (list): list of features name to drop from input dataframe
    
    Attributes:  
    '''

    def __init__(self, file_path, target, drop_features):
        '''Initialize the instance of the class with Args'''
        self.file_path = file_path
        self.target = target
        self.drop_features = drop_features

    def process_data(self):
        '''Processing of the input dataset'''
        df= pd.read_csv(self.file_path)

        # encode the target variable with value btw 0 to no of classes
        Label = df[self.target]
        le = preprocessing.LabelEncoder()
        y  = le.fit_transform(Label)
        
        # drop the selected features
        if self.drop_features is not None:
            X = df.drop(columns=self.drop_features)
            X = X.drop(columns=[self.target])
        else:
            X = df.drop(columns=[self.target])

        # for transforming data into uniform and normal distribution
        transformer = preprocessing.QuantileTransformer(random_state=0)

        # split the data into test and train set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
        X_train_trans = transformer.fit_transform(X_train)
        X_test_trans = transformer.transform(X_test)

        # convert back into dataframes
        X_train_trans = pd.DataFrame(X_train_trans, columns=X.columns)
        X_test_trans = pd.DataFrame(X_test_trans, columns=X.columns)

        clf = GradientBoostingClassifier(n_estimators=10)
        clf.fit(X_train_trans, y_train)
        y_predict = clf.predict(X_test_trans)
        accuracy_n0_ga = accuracy_score(y_test, y_predict)
        print("Accuracy: ", accuracy_n0_ga)

        return clf, X_train_trans, X_test_trans, y_train, y_test, accuracy_n0_ga

    def  run_GA(self, generations, population_size, 
                crossover_probability, max_features, outdir, 
                classifier, X_train_trans, X_test_trans,
                y_train, y_test):
                '''Implement the Genetic Algorithm for feature selection'''
                ga_estimator = GAFeatureSelectionCV(estimator = classifier,
                cv= 5,
                scoring= 'accuracy',
                population_size= population_size,
                generations= generations,
                n_jobs= -1,
                crossover_probability= crossover_probability,
                mutation_probability= 0.05,
                verbose= True,
                max_features= max_features,
                keep_top_k= 3,
                elitism= True
                )

                callback = ProgressBar()
                ga_estimator.fit(X_train_trans,y_train, callbacks=callback)
                features = ga_estimator.best_features_
                y_predict_ga = ga_estimator.predict(X_test_trans.iloc[:,features])
                accuracy = accuracy_score(y_test, y_predict_ga)
                print("Accuracy with GA: ", accuracy)
                final_selected_features = list(X_test_trans.iloc[:, features].columns)
                print("Final Selected Features: ", final_selected_features)

                ## save the final features into output dir (file)
                result_df = pd.read_csv(self.file_path).loc[:, final_selected_features]
                print("Saving the file in path: ", outdir)
                result_df.to_csv(outdir)