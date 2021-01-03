import pandas as pd
import numpy as np

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
        validation_split_date: datetime.date()
            first date to perform the splitting on.
            if not provided will set to be the minimum date in the data after the first training set
        date_column: string, deafult='record_date'
            date of each record
        gap: int, default=0
            for cases the test set does not come right after the train set,
            *gap* days are left between train and test sets
        
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