import numpy as np
import pandas as pd
import joblib
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from model_bulding import scale_data, split_data
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


def tune_parameter(model,parameter_grid, X_train_S, y_train, X_test_S , y_test):
    grid_search = GridSearchCV(
        estimator=model,
        param_grid= parameter_grid,
        cv=5,
        scoring='neg_mean_squared_error'

    )
    grid_search.fit( X_train_S, y_train)
    best_parms = grid_search.best_params_
    best_model=grid_search.best_estimator_
    train_score=grid_search.best_estimator_.score(X_train_S, y_train)
    test_score= grid_search.best_estimator_.score(X_test_S , y_test)
    model_performance={
        "train_score":train_score,
        "test_score":test_score
    }

    return best_parms, model_performance, best_model

def save_model(folder, best_params:dict, model_performance, best_model):
    joblib.dump(best_model, f"{folder}/model.pkl")

    with open(f"{folder}/hyperparameters.json", 'w') as fp:
        json.dump(best_params, fp)

    with open(f"{folder}/metrics.json", 'w') as fp:
        json.dump(model_performance, fp)   


if __name__ == "__main__":
    df=pd.read_csv('../data/processed/clean_data.csv')
    print(df)
    X,y = df.drop("median_house_value", axis=1), df["median_house_value"]
    X_train, X_test, y_train, y_test =train_test_split(X,y, test_size=0.2) 
    X_train_S, X_test_S =scale_data(X_train, X_test)
    # model =SGDRegressor()
    # parameter_grid={
    #     'alpha':[0.001, 0.01, 0.1],
    #     'max_iter': [1000, 800, 1200],
    #     'l1_ratio':[0.15, 0.1, 0.05]
    # }
    forest=RandomForestRegressor()

    # model=LinearRegression()
    # parameter_grid={

    # }
    parameter_grid={
    'n_estimators':[100,150, 200],
    'max_depth':[None,10,20,30],
    'min_samples_split':[5,10,20]
}

    best_params, model_performance, model_name = tune_parameter(forest, parameter_grid, X_train_S, y_train, X_test_S , y_test)
    print(best_params)
    print(model_performance)
    
    folder="results"
    os.makedirs(folder/{model_name})
   
    save_model(folder, best_params, model_performance, model_name)

