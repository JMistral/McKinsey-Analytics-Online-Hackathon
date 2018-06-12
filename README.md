# McKinsey-Analytics-Online-Hackathon
Source Code for McKinsey Analytics Online Hackathon
* Data Preprocessing: data cleaning, impute missing data. resample data due to imbalanced class
* Modeling: Multilayer Perceptron (Neural Network), performace evaluation with ROC curve and AUC

### Result so far:
1. MLP with simple imputation (median and mode) --> AUC=0.91
2. MLP with MICE imputation --> AUC = 0.96

### TODO
How can I improve the performance?
1. ~~Try data imputation with kNN or MICE~~(done)
2. Adjust resample method
3. Try another model (XGBoost, Random Forest etc.)
