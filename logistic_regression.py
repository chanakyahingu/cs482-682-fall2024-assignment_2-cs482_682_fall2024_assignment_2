import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
import argparse
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class MyLogisticRegression:
    def __init__(self, dataset_num, perform_test):
        self.training_set = None
        self.test_set = None
        self.model_logistic = None
        self.model_linear = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        
        self.perform_test = perform_test
        self.dataset_num = dataset_num
        self.read_csv(self.dataset_num)

    def read_csv(self, dataset_num):
        # Set the filenames for train and test datasets
        if dataset_num == '1':
            train_dataset_file = 'train_q1_1.csv'
            test_dataset_file = 'test_q1_1.csv'
        elif dataset_num == '2':
            train_dataset_file = 'train_q1_2.csv'
            test_dataset_file = 'test_q1_2.csv'
        else:
            print("Unsupported dataset number")
            return
        
        # Load the training set
        self.training_set = pd.read_csv(train_dataset_file, sep=',', header=0)
        self.X_train = self.training_set.iloc[:, :-1].values
        self.y_train = self.training_set.iloc[:, -1].values
        
        # Load the test set if perform_test is True
        if self.perform_test:
            self.test_set = pd.read_csv(test_dataset_file, sep=',', header=0)
            self.X_test = self.test_set.iloc[:, :-1].values
            self.y_test = self.test_set.iloc[:, -1].values

    def model_fit_linear(self):
        '''
        Initialize self.model_linear here and call the fit function
        '''
        self.model_linear = LinearRegression()
        self.model_linear.fit(self.X_train, self.y_train)
    
    def model_fit_logistic(self):
        '''
        Initialize self.model_logistic here and call the fit function
        '''
        self.model_logistic = LogisticRegression()
        self.model_logistic.fit(self.X_train, self.y_train)
    
    def model_predict_linear(self):
        '''
        Calculate and return the accuracy, precision, recall, f1, support of the model.
        '''
        # Fit the linear model
        self.model_fit_linear()
        accuracy = 0.0
        precision, recall, f1, support = np.array([0, 0]), np.array([0, 0]), np.array([0, 0]), np.array([0, 0])
        assert self.model_linear is not None, "Initialize the model, i.e. instantiate the variable self.model_linear in model_fit_linear method"
        assert self.training_set is not None, "self.read_csv function isn't called or the self.training_set hasn't been initialized."
        
        # Use test set if perform_test is True, otherwise use training set
        if self.perform_test:
            X = self.X_test
            y_true = self.y_test
        else:
            X = self.X_train
            y_true = self.y_train

        # Make predictions and convert to binary (0 or 1) using 0.5 threshold
        y_pred = self.model_linear.predict(X)
        y_pred_class = (y_pred >= 0.5).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred_class)
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred_class)
        
        assert precision.shape == recall.shape == f1.shape == support.shape == (2,), "precision, recall, f1, support should be an array of shape (2,)"
        return [accuracy, precision, recall, f1, support]

    def model_predict_logistic(self):
        '''
        Calculate and return the accuracy, precision, recall, f1, support of the model.
        '''
        # Fit the logistic model
        self.model_fit_logistic()
        accuracy = 0.0
        precision, recall, f1, support = np.array([0, 0]), np.array([0, 0]), np.array([0, 0]), np.array([0, 0])
        assert self.model_logistic is not None, "Initialize the model, i.e. instantiate the variable self.model_logistic in model_fit_logistic method"
        assert self.training_set is not None, "self.read_csv function isn't called or the self.training_set hasn't been initialized."
        
        # Use test set if perform_test is True, otherwise use training set
        if self.perform_test:
            X = self.X_test
            y_true = self.y_test
        else:
            X = self.X_train
            y_true = self.y_train

        # Make predictions
        y_pred = self.model_logistic.predict(X)

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred)
        
        assert precision.shape == recall.shape == f1.shape == support.shape == (2,), "precision, recall, f1, support should be an array of shape (2,)"
        return [accuracy, precision, recall, f1, support]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Linear Regression')
    parser.add_argument('-d', '--dataset_num', type=str, default="1", choices=["1", "2"], help='String indicating dataset number. For example, 1 or 2')
    parser.add_argument('-t', '--perform_test', action='store_true', help='Boolean to indicate inference (test data prediction)')
    args = parser.parse_args()

    classifier = MyLogisticRegression(args.dataset_num, args.perform_test)

    # Linear Regression Prediction
    acc = classifier.model_predict_linear()
    print(f"Linear Regression Results: {acc}")

    # Logistic Regression Prediction
    acc = classifier.model_predict_logistic()
    print(f"Logistic Regression Results: {acc}")
