import pandas as pd
import numpy as np

from sklearn.base import clone

from joblib import Parallel, delayed

#Code inspired by https://towardsdatascience.com/time-based-cross-validation-d259b13d42b8

class TimeSplitter(object):
    """
    CV splitter class for sklearn cross-validation. 
    """
    
    def __init__(self, train_period=500, test_period=100):
        '''
        Input:
            train_period: int
                Time units in the train set 
            test_period: int
                Time units in test set
        '''
        self.train_period = train_period
        self.test_period = test_period
        
        
    def split(self, data, validation_split_date=None, gap=0):
        '''
        Generate indices to split data into training and test set
        
        Parameters 
        ----------
        data: pandas DataFrame
            your data, contain one column for the record date 
        validation_split_date: 
            Date to first split from
            default is first timestamp for training data
        gap: int, default=0
            gap between train and test set, in time units
        
        Output:
            list of tuples (train index, test index)
        '''
                    
        train_indices_list = []
        test_indices_list = []

        if validation_split_date==None:
            validation_split_date = data.index[0] + self.train_period
        
        start_train = validation_split_date - self.train_period
        end_train = start_train + self.train_period
        start_test = end_train + gap
        end_test = start_test + self.test_period

        while end_test < data.index[-1]:
            # Test and train indices for current fold
            curr_train_indices = list(data[(data.index >= start_train) & (data.index < end_train)].index)
            curr_test_indices = list(data[(data.index >= start_test) & (data.index < end_test)].index)
            
            print("Train period:",start_train,"-" , end_train, ", Test period", start_test, "-", end_test,
                  "# train records", len(curr_train_indices), ", # test records", len(curr_test_indices))

            train_indices_list.append(curr_train_indices)
            test_indices_list.append(curr_test_indices)

            # update dates:
            start_train = start_train + self.test_period
            end_train = start_train + self.train_period
            start_test = end_train + gap
            end_test = start_test + self.test_period

        # Sklearn output formatting
        index_output = [(train,test) for train,test in zip(train_indices_list,test_indices_list)]

        self.n_splits = len(index_output)
        
        return index_output
    
    
    def get_n_splits(self):
        """ Getter for the amount of splitting iterations
        Output:
            n_splits : int
        """
        return self.n_splits 
    
# Number of jobs to run in parallel
N_JOBS = -1#10
N_SPLITS = 5
N_REPEATS = 4

def fit_and_score(estimator, X, y, train_index, test_index, df, optimize = False):
    y_train_fold = y[:,np.nonzero(np.in1d(df.index, train_index))[0]]
    y_test_fold = y[:,np.nonzero(np.in1d(df.index, test_index))[0]]
    estimator.fit(X, train_index, y_train_fold, optimize = optimize, fun = 'RMSE')
    score = estimator.score(X, test_index, y_test_fold, metric = "RMSE")
    
    return score
    
    
def cv_prediction(estimator, X, y, df, optimize = False):
    """
    Performs cross-validation on an estimator, spreading fitting and scoring over multiple jobs
    """
 
    n_jobs = N_JOBS
 
    splitter = TimeSplitter(1200, 240)
    split_index = splitter.split(df)
 
    parallel = Parallel(n_jobs=n_jobs)
    scores = parallel(
        delayed(fit_and_score)(
            clone(estimator), X, y, train_index, test_index, df, optimize
        ) for  train_index, test_index in split_index
    )
 
    return np.array(scores)
    
    
def cv_gridsearch(estimator, X, y, df, cv):
    return 0

    

def scorer(estimator, X, y):
    y_pred, _ = estimator.predict(X, timeInstants)

    y_true = targets[~np.isnan(targets)]
    y_pred = y_pred[~np.isnan(targets)]

    return mean_squared_error(y_true, y_pred, squared=False)