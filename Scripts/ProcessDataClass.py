import pandas as pd
import numpy as np


class ProcessData():

    """
    This class gets the model ready for machine learning implementation.

    """

    def __init__(self, df):
        self.df = df

    def calculate_half_life_per_subject(self,df):

        """
        This function calculates the half-life of a drug for a given subject. Given that we have the concentration of FVIII in the blood
        at three differe time points, we can calculate the half-life of the drug using the following formula:

        C = C_0 * e^(-kt)

        where:
            C = concentration at time t
            C_0 = initial concentration
            k = elimination rate constant
            t = time

        Taking the natural log of both sides of the equation, we get:

        ln(C) = ln(C_0) - kt

        We can then plot ln(C) against t to get a straight line with a slope of -k. The half-life of the drug can then be calculated using
        the following formula:

        t_1/2 = ln(2) / k

        To get k, we can use linear regression to find the slope of the line. The slope of the line is k, so we can then calculate the half-life

        Args:
            subject_data (pd.DataFrame): A dataframe containing the concentration of FVIII in the blood at different time points for a given subject

        Returns:
            float: The half-life of the drug for the given subject
        """
        
        times = df['time']
        concentrations = df['dv']

        # Logarithmic transformations of concentrations (ln(C) = ln(C_0) - kt)
        log_concentrations = np.log(concentrations)

        # Linear regression to find the slope (k)
        slope, intercept = np.polyfit(times, log_concentrations, 1)

        # calculate half-life
        half_life = np.log(2) / abs(slope)

        return half_life
    
    def add_half_life_column(self):
        """
        This function adds a column to the dataframe containing the half-life of the drug for each subject. The half-life of the drug is calculated
        using the calculate_half_life_per_subject method.
        """
        grouped = self.df.groupby('id')

        half_life = grouped.apply(self.calculate_half_life_per_subject)

        self.df['half_life'] = self.df['id'].map(half_life)

    
    def calculate_bmi(self, weight, height):
        """
        This function calculates the body mass index (BMI) of a given subject. The BMI is calculated using the following formula:
        BMI = weight / (height ** 2)

        Args:
            weight (float): The weight of the subject
            height (float): The height of the subject
        
        Returns:
            float: The BMI of the subject
        """
        
        # change the height from cm to m
        height = height / 100
        return weight / (height ** 2)
    
    def add_bmi_column(self):
        """
        This function adds a column to the dataframe containing the BMI of each subject. The BMI of the subject is calculated
        using the calculate_bmi method.
        """

        # Calculate the BMI for each subject
        self.df['bmi'] = self.calculate_bmi(self.df['wt'], self.df['ht'])
    
    def one_hot_encode(self, columns):
        """
        This function one-hot encodes the categorical variables, variables such as the FVIII mutations and product in the dataset.
        This is done to ensure that the categorical variables are in a format that can be used for machine learning model implementation.

        Args:
            df (pd.DataFrame): The dataframe containing the raw data
            columns (list): A list of the columns that need to be one-hot encoded
        """

        # One-hot encode the categorical variables with 1 for the presence of the category and 0 for the absence of the category
        self.df = pd.get_dummies(self.df, columns=columns, dtype=int)

    def group_by_subject(self):
        """
        This function groups the data by subject. This is done to ensure that the data is in a format that can be used for machine learning model implementation.
        """
        self.df = self.df.groupby('id').mean()

    def drop_uncessary_columns(self):
        """
        This function drops the columns that are not needed for machine learning model implementation.
        """
        self.df.drop(columns=['amt', 'duration','dv','mdv','time'], inplace=True)
    
    def drop_na(self):
        """
        This function drops the rows with missing values.
        """
        self.df.dropna(inplace=True)