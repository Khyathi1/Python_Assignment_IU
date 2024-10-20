Data Fitting and Mapping Using Least-Squares Method

Overview
This project demonstrates how to use the Least-Squares Method to fit a set of training data to ideal functions and map new test data to the best-fitting functions. The program performs the following tasks:

Data Loading: Reads training, test, and ideal function data from CSV files.
Least-Squares Fitting: Finds the best-fit functions for the training data from a set of 50 ideal functions.
Test Data Mapping: Maps new test data points to the closest matching ideal functions based on deviation.
Data Storage: Saves the results (training data, ideal functions, and mapped test data) to a SQLite database.
Data Visualization: Visualizes the relationships between training data, ideal functions, and test data.

Project Structure
train.csv: Contains the training data with x-y pairs for four datasets (Y1, Y2, Y3, Y4).
test.csv: Contains the test data with x-y pairs.
ideal.csv: Contains the ideal function data with x-y pairs for 50 functions.
Ideal_Function.py: The main Python script that runs the entire process.
functions.db: The SQLite database where the data is saved.

Explanation of the project:
1. Data Loading
The data is loaded from the CSV files using the Pandas library. Three datasets are loaded: training data, test data, and ideal functions.

2. Least-Squares Fitting
The Least-Squares Method is used to compare each training dataset to the ideal functions. The function that minimizes the sum of squared differences is selected as the best fit for each training dataset.

3. Test Data Mapping
The test data points are mapped to the closest ideal function by comparing their y-values to the y-values of the ideal functions and calculating the deviation.

4. Database Storage
The results (training data, ideal functions, and mapped test data) are saved in a SQLite database using the SQLAlchemy library.

5. Visualization
The data is visualized using Matplotlib, showing the training data, ideal functions, and test data.
