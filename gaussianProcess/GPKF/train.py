import numpy as np
from params import Params
from gpkfEstimation import Gpkf
from generateSyntheticData import generateSyntheticData

def start_train():
    params = Params()
    model = Gpkf(params)

    F, Y, noiseVar = generateSyntheticData(params.data)

    # GPKF estimate
    posteriorMean, posteriorCov, logMarginal = model.estimation(Y, noiseVar)

    #GPKF prediction
    predictedMean, predictedCov = model.prediction(Y, noiseVar)

    return model

def main():
    
    # Load dataFrame and format index to desired formaat (0 - T)
    df = pd.read_csv('../../data/mergedData.csv')
    df.set_index('Timestamp', inplace=True)
    df.index = pd.to_datetime(df.index)
    df = df.asfreq('1Min')
    df.index = np.around(df.index.to_julian_date() * 24 * 60)
    df.index = df.index - df.index[0]
    
    X_train, X_test, y_train, y_test = train_test_split(df_do.index.values, df_do.values, test_size=0.25, shuffle= False)
    
    # Format data: one row for each measurement location, column for time
    Y = df_do.to_numpy().T
    Y_train = Y[:,:len(X_train)]
    Y_train_timeInstants = df_do.index[:len(X_train)]
    Y_test = Y[:,len(X_train):]
    Y_test_timeInstants = df_do.index[len(X_train):]
    
    
    print('Starting training')
    model = start_train()
    print('Finished training')

if __name__ == '__main__':
    main()