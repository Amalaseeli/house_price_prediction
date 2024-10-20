
# House price prediction using regression techniques

This project involves building and evaluating different regression models to predict housing prices using various machine learning algorithms. The models include Stochastic Gradient Descent (SGD) Regressor, Random Forest Regressor, Decision Tree Regressor, and Gradient Boosting Regressor.


## Features

- Multiple regression models implemented:
  - Stochastic Gradient Descent (SGD) Regressor
  - Random Forest Regressor
   - Decision Tree Regressor
  - Gradient Boosting Regressor
- Hyperparameter tuning using Grid Search
- Model performance visualization through R² score 
## Technologies Used

**Programming Language:**  
- Python

**Libraries:** 
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib




## Installation

Clone the repository to your local machine

```bash
 git clone https://github.com/Amalaseeli/house_price_prediction.git
```
    
## Model Evaluation

### Model Performance Summary

The table below summarizes the performance of various regression models evaluated in this project. The scores are based on the R² metric, which indicates how well the models fit the data.

| Model                         | Train R² Score | Test R² Score |
|-------------------------------|----------------|----------------|
| Stochastic Gradient Descent   | 0.62          | 0.62        |
| Random Forest Regressor       | 0.90           | 0.73         |
| Decision Tree Regressor       | 0.59          | 0.59          |
| **Gradient Boosting Regressor**   | **0.81**          | **0.74**         |

### Evaluation Metrics

- The performance of the models is evaluated using R² scores on training and test datasets. 
- The model with the highest test R² score is considered the best-performing model.

### Best Model

The **Gradient Boosting Regressor** has been identified as the best model for this regression problem based on its superior performance in terms of both training and test R² scores. 

### Evaluation Metrics

- The performance of the models is evaluated using R² scores on training and test datasets. 
- The model with the highest test R² score is considered the best-performing model.

### Conclusion

The **Gradient Boosting Regressor** provides a robust solution for the regression task, achieving a test R² score of **0.74**. Further tuning of its hyperparameters may enhance its performance even more.
