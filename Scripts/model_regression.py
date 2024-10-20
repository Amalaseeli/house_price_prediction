import pandas as pd
import numpy as np
import joblib
import json
import os
from hyperparameter import tune_parameter, save_model
from model_building import scale_data, split_data
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor


model_list=[
    SGDRegressor(),
    RandomForestRegressor(),
    DecisionTreeRegressor(),
    GradientBoostingRegressor()

]

#SGD Regressor
parameter_sgd={
    "alpha":[0.0001, 0.001],
    "max_iter": [1000, 800, 1100],
    "learning_rate":['invscaling', 'optimal', 'adaptive']
}

#Random Forest Regressor
parameter_rfr={
    'n_estimators':[100,150, 200],
    'max_depth':[None,10,20,30],
    'min_samples_split':[5,10,20]
}

#Decision Tree Regressor
parameter_dtr={
    "max_depth":[2,3,4],
    "min_samples_split":[2,3,4],
    "min_samples_leaf":[1,2,3,4],
    "splitter" : ["best", "random"]
}

#Gradient Boost Regressor
parameter_gbr={
    "learning_rate":[0.1, 0.01, 0.001],
    "n_estimators": [100, 200, 300],
    "min_samples_split":[2, 5, 10],
    "max_depth":[2,3,4]
}
parameter_grid_list=[parameter_sgd,parameter_rfr, parameter_dtr, parameter_gbr]

def evaluate_models(model_list, parameter_grid_list, X_train_S, y_train):
    for i in range(len(model_list)):
        model=model_list[i]
        parameter_grid=parameter_grid_list[i]

        best_params, model_performance, best_model= tune_parameter(model,parameter_grid, X_train_S, y_train, X_test_S , y_test)

        model_name=str(model)[:-2]
        folder=f"results/{model_name}"
        os.makedirs(folder, exist_ok=True)

        save_model(folder, best_params, model_performance, best_model)

def find_best_model(model_list):
    dict={}

    for model in model_list:
        model_name=str(model)[:-2]
        metric_files=f"results/{model_name}/metrics.json"

        with open (metric_files) as f:
            metric=json.load(f)
        
        dict[model_name] = metric["test_score"]

    best_model_name=max(dict, key=dict.get)

    with open(f"results/{best_model_name}/hyperparameters.json") as json_file:
                parameters = json.load(json_file)

    with open(f"results/{best_model_name}/metrics.json") as json_file:
                performance_metric = json.load(json_file)

    best_reg_model = joblib.load(f"results/{best_model_name}/model.pkl")
    return best_reg_model, parameters, performance_metric
    

    
if __name__=="__main__":
    df=pd.read_csv('../data/processed/clean_data.csv')
    X,y = df.drop("median_house_value", axis=1), df["median_house_value"]
    X_train , X_test, y_train, y_test=split_data(X,y)
    print(X_train.shape)
    print(y_train.shape)
    X_train_S, X_test_S=scale_data(X_train, X_test)

    evaluate_models(model_list, parameter_grid_list, X_train_S, y_train)
    best_reg_model, parameters, performance_metric = find_best_model(model_list)
    
    print(f"The best regression model is {best_reg_model}")
    print(f"Performance metric for the best model is {performance_metric}")
