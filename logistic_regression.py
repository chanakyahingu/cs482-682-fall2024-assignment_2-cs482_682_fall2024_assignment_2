import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.model_selection import train_test_split

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
        if dataset_num == '1':
            train_dataset_file = 'train_q1_1.csv'
            test_dataset_file = 'test_q1_1.csv'
        elif dataset_num == '2':
            train_dataset_file = 'train_q1_2.csv'
            test_dataset_file = 'test_q1_2.csv'
        else:
            print("unsupported dataset number")
            
        self.training_set = pd.read_csv(train_dataset_file, sep=',', header=0)
        if self.perform_test:
            self.test_set = pd.read_csv(test_dataset_file, sep=',', header=0)
        
        
    def model_fit_linear(self):
        '''
        initialize self.model_linear here and call the fit function
        '''
        X = self.training_set[['exam_score_1', 'exam_score_2']]  # Features (two exam scores)
        y = self.training_set['label']  # Target (admission decision)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model_linear = LinearRegression()
        self.model_linear.fit(self.X_train, self.y_train)
    
    def model_fit_logistic(self):
        '''
        initialize self.model_logistic here and call the fit function
        '''
        X = self.training_set[['exam_score_1', 'exam_score_2']]  # Features (two exam scores)
        y = self.training_set['label']  # Target (admission decision)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model_logistic = LogisticRegression()
        self.model_logistic.fit(self.X_train, self.y_train)
        
    
    def model_predict_linear(self):
        '''
        Calculate and return the accuracy, precision, recall, f1, support of the model.
        '''
        self.model_fit_linear()
        accuracy = 0.0
        precision, recall, f1, support = np.array([0,0]), np.array([0,0]), np.array([0,0]), np.array([0,0])
        assert self.model_linear is not None, "Initialize the model, i.e. instantiate the variable self.model_linear in model_fit_linear method"
        assert self.training_set is not None, "self.read_csv function isn't called or the self.trianing_set hasn't been initialized "
        
        if self.X_test is not None:
            y_pred = self.model_linear.predict(self.X_test)
            # perform prediction here
            y_pred_class = (y_pred >= 0.5).astype(int)
            precision, recall, f1, support = precision_recall_fscore_support(self.y_test, y_pred_class, average=None)
            
        assert precision.shape == recall.shape == f1.shape == support.shape == (2,), "precision, recall, f1, support should be an array of shape (2,)"
        return [accuracy, precision, recall, f1, support]

    def model_predict_logistic(self):
        '''
        Calculate and return the accuracy, precision, recall, f1, support of the model.
        '''
        self.model_fit_logistic()
        accuracy = 0.0
        precision, recall, f1, support = np.array([0,0]), np.array([0,0]), np.array([0,0]), np.array([0,0])
        assert self.model_logistic is not None, "Initialize the model, i.e. instantiate the variable self.model_logistic in model_fit_logistic method"
        assert self.training_set is not None, "self.read_csv function isn't called or the self.trianing_set hasn't been initialized "
        if self.X_test is not None:
            y_pred = self.model_logistic.predict(self.X_test)
            precision, recall, f1, support = precision_recall_fscore_support(self.y_test, y_pred, average=None)
            # perform prediction here
        assert precision.shape == recall.shape == f1.shape == support.shape == (2,), "precision, recall, f1, support should be an array of shape (2,)"
        return [accuracy, precision, recall, f1, support]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Linear Regression')
    parser.add_argument('-d','--dataset_num', type=str, default = "1", choices=["1","2"], help='string indicating datset number. For example, 1 or 2')
    parser.add_argument('-t','--perform_test', action='store_true', help='boolean to indicate inference')
    args = parser.parse_args()
    classifier = MyLogisticRegression(args.dataset_num, args.perform_test)
    acc = classifier.model_predict_linear()
    acc = classifier.model_predict_logistic()
    