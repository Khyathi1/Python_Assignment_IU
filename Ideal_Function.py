import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import matplotlib.pyplot as plt

# Load the training, test, and ideal function datasets
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
ideal_data = pd.read_csv('ideal.csv')

# Display the first few rows of each dataset to understand its structure
print("Training Data:")
print(train_data.head())

print("Test Data:")
print(test_data.head())

print("Ideal Functions Data:")
print(ideal_data.head())



def least_squares(train_y, ideal_y):
    # Calculate the sum of squared differences between the training y-values and ideal y-values
    return np.sum((train_y - ideal_y) ** 2)

# Initialize a list to store the best fit ideal function for each training dataset
best_fit_functions = []

# Loop through the 4 training datasets (Y1, Y2, Y3, Y4)
for i in range(1, 5):  # For each training function Y1, Y2, Y3, Y4
    best_fit = None
    min_error = float('inf')
    
    # Compare the current training function with all 50 ideal functions (Y1 to Y50)
    for j in range(1, 51):  # There are 50 ideal functions
        train_y = train_data[f'y{i}']
        ideal_y = ideal_data[f'y{j}']
        
        # Ensure train_y and ideal_y have no missing values
        if train_y.notnull().all() and ideal_y.notnull().all():
            # Calculate the least-squares error
            error = least_squares(train_y, ideal_y)
            
            # If this ideal function has a smaller error, update the best fit
            if error < min_error:
                min_error = error
                best_fit = j  # Store the ideal function index
    
    # Append the best fit ideal function for this training dataset
    best_fit_functions.append(best_fit)

print(f"Best fit functions for training datasets: {best_fit_functions}")

def map_test_to_ideal(test_data, ideal_data, best_fit_functions):
    results = []

    # Loop through each test data point
    for index, row in test_data.iterrows():
        x_test, y_test = row['x'], row['y']
        
        # Iterate over the four best-fit ideal functions
        for i,ideal_func_idx in enumerate(best_fit_functions):
            # Get the y-value from the ideal function corresponding to this x value
            ideal_y = ideal_data.loc[ideal_data['x'] == x_test, f'y{ideal_func_idx}'].values[0]
            
            # Calculate the deviation between the test data y and the ideal function y
            deviation = abs(y_test - ideal_y)
            
            # Store the result (x, y, deviation, ideal function number)
            results.append((x_test, y_test, deviation, ideal_func_idx))
    
    # Convert the results into a DataFrame for easy handling
    return pd.DataFrame(results, columns=['x', 'y', 'Deviation', 'Ideal Function'])

# Map the test data to the ideal functions
mapped_test_data = map_test_to_ideal(test_data, ideal_data, best_fit_functions)

# Check the mapped test data
print("Mapped Test Data:")
print(mapped_test_data.head())



# Create a SQLite engine
engine = create_engine('sqlite:///functions.db', echo=False)

# Save the data into tables in the SQLite database
train_data.to_sql('training_data', con=engine, if_exists='replace', index=False)
ideal_data.to_sql('ideal_functions', con=engine, if_exists='replace', index=False)
mapped_test_data.to_sql('mapped_test_data', con=engine, if_exists='replace', index=False)

print("Data saved to SQLite database 'functions.db'")



def visualize(train_data, ideal_data, mapped_test_data, best_fit_functions):
    plt.figure(figsize=(10, 6))

    # Plot the training data
    for i in range(1, 5):
        plt.scatter(train_data['x'], train_data[f'y{i}'], label=f'Training y{i}', s=10)
    
    # Plot the best-fit ideal functions
    for i, ideal_func_idx in enumerate(best_fit_functions):
        plt.plot(ideal_data['x'], ideal_data[f'y{ideal_func_idx}'], label=f'Ideal Function {ideal_func_idx}', linestyle='--')

    # Plot the mapped test data
    plt.scatter(mapped_test_data['x'], mapped_test_data['y'], label='Test Data', c='red', s=20, marker='x')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('Training Data, Ideal Functions, and Test Data')
    plt.show()

# Visualize the data
visualize(train_data, ideal_data, mapped_test_data, best_fit_functions)
