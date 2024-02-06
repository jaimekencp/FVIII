import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import os

class ImplementMachineLearning():
    def __init__(self, df):
        self.df = df
    

    def find_correlated_features(self):
        """
        This function removes features that are highly correlated with each other. 
        This is done to prevent overfitting of the model.
        """

        # excluding the target variable 'half_life'
        df = self.df.drop(columns='half_life')

        corr_matrix = df.corr().abs()

        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Find features with correlation greater than 0.95
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

        # Drop features 
        self.df.drop(to_drop, axis=1, inplace=True)

    def split_data(self):
        """
        This function splits the data into training and testing sets.
        """

        X = self.df.drop(columns='half_life')
        y = self.df['half_life']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test
    
    def linear_regression(self, X_train, X_test, y_train, y_test):
        """
        This function fits a linear regression model to the data and returns the mean squared error and R^2 score.
        """

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        model_coefs = model.coef_

        return mse, r2, model_coefs
    
    def train_models(self, model, X_train, y_train, hyperparameters):
        """
        This function tunes the hyperparameters of the models and trains them
        using a randomized 5 fold cross-validation.
        """

        grid_search = GridSearchCV(model, hyperparameters, cv=5, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_

        return model
    
    def random_forrest_regressor(self, X_train, X_test, y_train, y_test):

        model_random_forest = RandomForestRegressor()

        # random forrest hyperparameters
        hyperparameters = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30]
        }

        trained_random_forest_model = self.train_models(model_random_forest, X_train, y_train, hyperparameters)

        y_pred = trained_random_forest_model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        model_feature_importances = trained_random_forest_model.feature_importances_

        return mse, r2, model_feature_importances

    def gradient_boosting_regressor(self, X_train, X_test, y_train, y_test):

        model_gradient_boosting = GradientBoostingRegressor()

        # gradient boosting hyperparameters
        hyperparameters = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30]
        }

        trained_gradient_boosting_model = self.train_models(model_gradient_boosting, X_train, y_train, hyperparameters)

        y_pred = trained_gradient_boosting_model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        model_feature_importances = trained_gradient_boosting_model.feature_importances_

        return mse, r2, model_feature_importances
    
    def plot_feature_importances(self, X_train, model_feature_importance, model_name):
        """
        This function plots the feature importances of the models.
        """

        if model_name in ['Random Forest', 'Gradient Boosting']:
            importances = model_feature_importance
            feature_names = X_train.columns
            importances_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
            importances_df = importances_df.sort_values(by='Importance', ascending=False)
            
            plt.figure(figsize=(10, 5))
            sns.barplot(x='Importance', y='Feature', data=importances_df)
            plt.title(f"Feature Importance of {model_name} for half-life")

            # create a folder inside Plots and name it after the model and target
            folder_name = f"{model_name}_half-life"
            os.makedirs(os.path.join("Data_output", folder_name), exist_ok=True)

            plt.tight_layout()

            # save the plot inside the folder
            plt.savefig(os.path.join("Data_output", folder_name, f'{model_name}_half-life_Importance.png'))

        else:
            model_feature_importance
            feature_names = X_train.columns
            coefficients_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': model_feature_importance})
            coefficients_df = coefficients_df.sort_values(by='Coefficient', ascending=False)
            
            plt.figure(figsize=(10, 5))
            sns.barplot(x='Coefficient', y='Feature', data=coefficients_df)
            plt.title(f"Feature Coefficients for half-life using {model_name}")
            plt.show()

            # create a folder inside Plots and name it after the model and target
            folder_name = f"{model_name}_half-life"
            os.makedirs(os.path.join("Data_output", folder_name), exist_ok=True)

            plt.tight_layout()

            # save the plot inside the folder
            plt.savefig(os.path.join("Data_output", folder_name, f'{model_name}_half_life_coefficients.png'))

    def plot_mse(self, model_mse_dict):
        """
        This function plots the mean squared error of the models.
        """

        model_mse_df = pd.DataFrame(model_mse_dict.items(), columns=['Model', 'Mean Squared Error'])
        plt.figure(figsize=(10, 5))
        sns.barplot(x='Model', y='Mean Squared Error', data=model_mse_df)
        plt.title('Mean Squared Error of Models')
        plt.show()

        # save the plot
        plt.savefig('Data_output/mse_plot.png')

    def plot_r2(self, model_r2_dict):
        """
        This function plots the R^2 score of the models.
        """

        model_r2_df = pd.DataFrame(model_r2_dict.items(), columns=['Model', 'R^2'])
        plt.figure(figsize=(10, 5))
        sns.barplot(x='Model', y='R^2', data=model_r2_df)
        plt.title('R^2 of Models')
        plt.show()

        # save the plot
        plt.savefig('Data_output/r2_plot.png')