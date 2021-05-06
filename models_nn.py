import os
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Flatten, Conv1D, MaxPooling1D
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.optimizers import SGD, Adam
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
import tensorflow as t

PATH = "..."
SIZE_SAMPLE = 500000

def get_database(path, sample_size=None):
    datasets = list()
    for i in range(10):
        file_ = f"base_{i}.pkl"
        filepath = os.path.join(path, file_)
        base = pd.read_pickle(filepath)
        datasets.append(base)
    base = pd.concat(datasets)
    base.to_csv('./base.csv', sep=';')

    if sample_size is not None:
        base = base.sample(n=sample_size)

    base = base.sort_values(['X4'], ascending=True).reset_index(drop=True) 

    flag_columns = base.columns[pd.Series(base.columns).str.startswith('Flag_')]

    for fc in flag_columns:
        base[fc] = base[fc].astype(int)
    
    return base

def prepare_train_test(X, Y):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    scaler = StandardScaler()
    
    X_train = scaler.fit_transform(X_train)

    X_test = scaler.fit_transform(X_test)

    return X_train, y_train, X_test, y_test

def X_Y_database(base):
    X = base.drop(columns=['X8', 'X1', 'X2', 'X4']) * 1
    y = base.X8 * 1
    
    return X.values, y.values

def create_larger(dropout=0.4):
	# create model
    model = Sequential()
    
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', strides=1, input_shape=(37,1)))    
    model.add(Dropout(dropout))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
        
    #output layer
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    model.compile(optimizer=Adam(0.0005),
                loss='binary_crossentropy',
                metrics=['AUC'])

    print(model.summary())

    return model

if __name__ == '__main__':
    dataset = get_database(PATH)
    x, y = X_Y_database(dataset)

    X_train, y_train, X_test, y_test = prepare_train_test(x, y)

    model = create_larger()

    model.fit(
        X_train.reshape(X_train.shape[0], X_train.shape[1], 1), y_train, 
        batch_size=128,
        epochs=100, 
        validation_split=0.1,
        verbose=1)

    model.save('./model.h5')

    y_pred = model.predict(X_test.reshape(X_test.shape[0], X_test.shape[1], 1))
    print(y_pred)

    pd.DataFrame({'pred': y_pred.reshape(y_pred.shape[0]), 'y_test': y_test.tolist()}).to_csv('./pred_test.csv', sep=';')


    roc = roc_auc_score(y_test, y_pred)

    print(roc)