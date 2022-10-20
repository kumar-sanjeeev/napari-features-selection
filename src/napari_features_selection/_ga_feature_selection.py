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


class GA:
    """This class contains the methods to implement the pipeline for implementing Genetic Algorithm for Feature Selection"""

    def __init__(self, dataset, target, drop_features):
        """
        Initilaize the dataset on which GA algo will be implemented

        Args:
            :param dataset(.csv file): input dataset from the selected csv file
            :param target(str): feature to become target, selected by user from GUI
            :param drop_features(list): list of features to drop, selected by user from GUI

        Returns:
            None
        """
        self.dataset = dataset
        self.target = target
        self.drop_features = drop_features

    def process_data(self):
        """
        Method resposible for doing pre-processing on the input dataset

        Args:
            None

        Returns:
            param X(dataframe): Input features
            param y(dataframe): Target label
            param cols_to_drop(list): List of features to drop
        """
        # encoding the target labels with value btw 0 to no_classes
        # learn about this : https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
        Label = self.dataset[self.target]
        le = preprocessing.LabelEncoder()
        y = le.fit_transform(Label)

        if self.drop_features is not None:
            cols_to_drop = self.drop_features
            # removing the selected features from the dataset and also the target feature
            X = self.dataset.drop(columns=cols_to_drop)
            X = X.drop(columns=[self.target])
        else:
            X = self.dataset.drop(columns=[self.target])

        return X, y, cols_to_drop

    def split_transform_data(X, y):
        # use for transforming the features to follow a uniform and normal distribution
        quantile_transformer = preprocessing.QuantileTransformer(
            random_state=0)

        # test and train split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=0)
        X_train_trans = quantile_transformer.fit_transform(X_train)
        X_test_trans = quantile_transformer.fit_transform(X_test)

        # convert back to dataframes
        X_train_trans = pd.DataFrame(X_train_trans, columns=X.columns)
        X_test_trans = pd.DataFrame(X_test_trans, columns=X.columns)

        # classifier used in GA
        clf = GradientBoostingClassifier(n_estimators=10)
        clf.fit(X_train_trans, y_train)
        y_predict = clf.predict(X_test_trans)
        accuracy_no_GA = accuracy_score(y_test, y_predict)
        print(
            f"Accuracy score of Classification without GA Features selection: {accuracy_no_GA:.2f}")

        return clf, X_train_trans, X_test_trans, y_test, y_train, accuracy_no_GA

    def run_GA(generations,
               population_size,
               crossover_prob,
               max_features,
               outdir, clf,
               X_train_trans,
               X_test_trans,
               y_test,
               y_train,
               accuracy_no_GA,
               drop_features):

        evolved_estimator = GAFeatureSelectionCV(estimator=clf,
                                                 cv=5,
                                                 scoring='accuracy',
                                                 population_size=population_size,
                                                 generations=generations,
                                                 n_jobs=1,
                                                 crossover_probability=crossover_prob,
                                                 mutation_probability=0.05,
                                                 verbose=True,
                                                 max_features=max_features,
                                                 keep_top_k=3,
                                                 elitism=True)

        callback = ProgressBar()
        evolved_estimator.fit(X_train_trans, y_train, callbacks=callback)
        features = evolved_estimator.best_features_
        y_predict_ga = evolved_estimator.predict(X_test_trans.iloc[:, features])
        accuracy = accuracy_score(y_test, y_predict_ga)
        print(accuracy)
        print(evolved_estimator.best_features_)


        plt.figure()
        plot = plot_fitness_evolution(evolved_estimator, metric='fitness')
        plt.savefig('fitness.png')
        selected_features = list(X_test_trans.iloc[:,features].columns)
        cv_results = evolved_estimator.cv_results_
        history = evolved_estimator.history


        generations = int(generations)
        population_size = int(population_size)
        crossover_prob = float(crossover_prob)

        if max_features is not None:
            print(f"Max features has been set to : {max_features}")
            max_features = int(max_features)
        else:
            max_features = None
            print(f"Max features has not been set")

        
        start_time = time.time()
        hr_start_time = datetime.datetime.fromtimestamp(start_time).strftime('%Y%m%d_%H-%M-%S')

        print("Running main function")

        results_df = pd.DataFrame(cv_results)
        results_df.to_csv('results_df.csv')

        # history_df = pd.DataFrame(history)
        # history_df.to_csv('history_df.csv')

        # plt.figure()
        # sns.violinplot(data=history_df.iloc[:,1:])
        # plt.savefig('history_result.png')

        # pd.DataFrame(selected_features).to_csv(str(hr_start_time) +'_selected_features.csv')     


        return selected_features, plot  