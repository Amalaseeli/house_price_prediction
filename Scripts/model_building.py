import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score 
from sklearn.metrics import root_mean_squared_error

def split_data(X,y):
    X_train , X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
    return X_train, X_test, y_train, y_test

def scale_data(X_train,X_test):
    scale=StandardScaler()
    X_train_S=scale.fit_transform(X_train)
    X_test_S=scale.fit_transform(X_test)
    return X_train_S, X_test_S

def predict_output(model,X_train_S,X_test_S):
    
    model.fit(X_train_S,y_train)
    y_train_predict=model.predict(X_train_S)
    y_test_predict=model.predict(X_test_S)
    return y_train_predict, y_test_predict


def calculate_r2_score(y_train_predict, y_test_predict, y_train, y_test):
    train_r2_score=r2_score(y_train_predict,y_train)
    test_r2_score=r2_score(y_test_predict,y_test)
    return train_r2_score, test_r2_score

def calculate_rmse(y_train_predict, y_test_predict, y_train, y_test):
    train_rmse=float(root_mean_squared_error(y_train_predict,y_train))
    test_rmse=float(root_mean_squared_error(y_test_predict,y_test))
    return train_rmse, test_rmse

if __name__ == "__main__":
    df=pd.read_csv('../data/processed/clean_data.csv')
    X,y = df.drop("median_house_value", axis=1), df["median_house_value"]
    print(X.shape)
    print(y.shape)
    X_train, X_test, y_train, y_test = split_data(X,y)
    X_train_S, X_test_S= scale_data(X_train,X_test)
    model = LinearRegression()
    y_train_predict, y_test_predict = predict_output(model,X_train_S, X_test_S)
    train_r2_score, test_r2_score = calculate_r2_score(y_train_predict, y_test_predict, y_train, y_test)
    train_rmse, test_rmse = calculate_rmse(y_train_predict, y_test_predict, y_train, y_test)
    print(f"train_r2_score:{train_r2_score},  test_r2_score: {test_r2_score}")
    print(f"train_rmse:{train_rmse},  test_rmse: {test_rmse}")
    

    


