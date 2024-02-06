import pandas as pd
from ProcessDataClass import ProcessData
from MachineLearningClass import ImplementMachineLearning

def process_data_for_modeling(df):

    # for all rows where 'dv' is na remove it from the dataframe
    df = df.dropna(subset=['dv'])

    process_data = ProcessData(df)

    process_data.add_half_life_column()

    process_data.add_bmi_column()

    process_data.one_hot_encode(columns=['product', 'mutation'])

    process_data.drop_uncessary_columns()

    process_data.group_by_subject()

    process_data.drop_na()

    return process_data.df

def machine_learning_models(df):

    df_implement_machine_learning = ImplementMachineLearning(df)

    df_implement_machine_learning.find_correlated_features()

    X_train, X_test, y_train, y_test = df_implement_machine_learning.split_data()

    # Linear Regression
    linear_mse, linear_r2, linear_model_coefs = df_implement_machine_learning.linear_regression(X_train, X_test, y_train, y_test)

    # Random Forest
    random_forest_mse, random_forest_r2, random_forest_model_feature_importance = df_implement_machine_learning.random_forrest_regressor(X_train, X_test, y_train, y_test)

    # Gradient Boosting
    gradient_boosting_mse, gradient_boosting_r2, gradient_boosting_model_feature_importance = df_implement_machine_learning.gradient_boosting_regressor(X_train, X_test, y_train, y_test)

    model_mse_dict = {'Linear Regression': linear_mse, 'Random Forest': random_forest_mse, 'Gradient Boosting': gradient_boosting_mse}
    df_implement_machine_learning.plot_mse(model_mse_dict)

    model_r2_dict = {'Linear Regression': linear_r2, 'Random Forest': random_forest_r2, 'Gradient Boosting': gradient_boosting_r2}
    df_implement_machine_learning.plot_r2(model_r2_dict)

    # plot feature importance for all models
    df_implement_machine_learning.plot_feature_importances(X_train, random_forest_model_feature_importance, 'Random Forest')
    df_implement_machine_learning.plot_feature_importances(X_train, gradient_boosting_model_feature_importance, 'Gradient Boosting')
    df_implement_machine_learning.plot_feature_importances(X_train, linear_model_coefs, 'Linear Regression')

  
def main():
    df = pd.read_csv('Raw_data/data.csv')
    df = process_data_for_modeling(df)
    machine_learning_models(df)

if __name__ == '__main__':
    main()