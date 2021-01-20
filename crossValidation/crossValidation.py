import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss, r2_score

# Number of jobs to run in parallel
N_JOBS = 10
N_SPLITS = 5
N_REPEATS = 4

def fit_and_score(estimator, X, timeInstants, y, train_index, test_index):
    y_train_fold = y[:,np.nonzero(np.in1d(df_do.index, train_index))[0]]
    y_test_fold = y[:,np.nonzero(np.in1d(df_do.index, test_index))[0]]
    estimator.fit(locations_values, y_train_timeInstants, y_train_fold)
    scores.append(model.score(locations_values, y_test))
    #y_eval = np.concatenate((y_train_fold, np.full(y_test_fold.shape, np.nan)), axis=1)
    #scores.append(model.score(locations_values, y_eval))
    
    
def repeated_k_fold(estimator, X, y):
    """
    Performs cross-validation on an estimator, spreading fitting and scoring over multiple jobs
    """
 
    n_jobs = N_JOBS
 
    n_splits = N_SPLITS
    n_repeats = N_REPEATS
 
 
    _k_fold = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats)
    splitter = TimeSplitter(1200, 240)
    split_index = splitter.split(df_do.iloc[:len(X_train)])
 
    parallel = Parallel(n_jobs=n_jobs)
    scores = parallel(
        delayed(_fit_and_score)(
            clone(estomator), X, y, w, train_index, test_index, i
        ) for i, (train_index, test_index) in enumerate(_k_fold.split(X))
    )
 
    return np.array(scores)
    
    

def scorer(estimator, X, y):
    y_pred, _ = estimator.predict(X, timeInstants)

    y_true = targets[~np.isnan(targets)]
    y_pred = y_pred[~np.isnan(targets)]

    return mean_squared_error(y_true, y_pred, squared=False)