import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.externals import joblib
import xgboost as xgb

# Step 1: Data Generation
# Create a synthetic dataset with 1 million rows
def generate_data():
    # Number of rows in the dataset
    num_rows = 1000000  # 1 million rows

    # Generate synthetic data
    customer_ids = np.random.randint(1, 50001, size=num_rows)  # 50,000 unique customers
    ages = np.random.randint(18, 70, size=num_rows)  # Age between 18 and 70
    genders = np.random.choice(['Male', 'Female'], size=num_rows)  # Random Male/Female
    product_ids = np.random.randint(1001, 1100, size=num_rows)  # 100 products (Product IDs 1001-1100)
    purchase_amounts = np.round(np.random.uniform(5.0, 500.0, size=num_rows), 2)  # Random purchase amount
    purchase_dates = [datetime.now() - timedelta(days=np.random.randint(0, 365)) for _ in range(num_rows)]  # Random dates within the last year

    # Create DataFrame
    df = pd.DataFrame({
        'customer_id': customer_ids,
        'age': ages,
        'gender': genders,
        'product_id': product_ids,
        'purchase_amount': purchase_amounts,
        'purchase_date': purchase_dates
    })

    # Save the dataset to CSV
    df.to_csv('customer_purchase_data.csv', index=False)
    print("Dataset generated and saved to 'customer_purchase_data.csv'.")

# Step 2: Feature Engineering
# Let's create new features based on spending behavior and frequency of purchase
def feature_engineering(df):
    # Extract the month and day of the week from the purchase date
    df['purchase_date'] = pd.to_datetime(df['purchase_date'])
    df['purchase_month'] = df['purchase_date'].dt.month
    df['purchase_day'] = df['purchase_date'].dt.dayofweek
    
    # Feature: Spending Category (Low, Medium, High)
    df['spending_category'] = pd.cut(df['purchase_amount'], bins=[0, 50, 150, 500], labels=['Low', 'Medium', 'High'])
    
    # Feature: Frequency of purchase (number of purchases in the last 30 days)
    df['purchase_frequency'] = df.groupby('customer_id')['purchase_date'].transform(lambda x: (datetime.now() - x.max()).days)
    
    # Feature: Average purchase amount per customer
    df['avg_purchase_amount'] = df.groupby('customer_id')['purchase_amount'].transform('mean')
    
    return df

# Step 3: Model Preparation and Pipeline
def prepare_model(df):
    # Selecting features and target
    X = df[['age', 'gender', 'product_id', 'purchase_month', 'purchase_day', 'spending_category', 'purchase_frequency', 'avg_purchase_amount']]
    y = df['purchase_amount']
    
    # Splitting dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Column transformer for preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['age', 'product_id', 'purchase_month', 'purchase_day', 'purchase_frequency', 'avg_purchase_amount']),
            ('cat', OneHotEncoder(), ['gender', 'spending_category'])
        ])

    # Models to compare: Linear Regression, Random Forest, and XGBoost
    models = {
        'Linear Regression': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ]),
        'Random Forest': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ]),
        'XGBoost': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', xgb.XGBRegressor(n_estimators=100, random_state=42))
        ])
    }

    return X_train, X_test, y_train, y_test, models

# Step 4: Model Training, Evaluation, and Hyperparameter Tuning
def train_and_evaluate(models, X_train, X_test, y_train, y_test):
    for name, model in models.items():
        print(f"\nTraining and evaluating {name} model...")
        
        # Cross-validation
        cv_score = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        print(f"Cross-Validation Score (MSE): {np.mean(cv_score)}")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Predict on test data
        y_pred = model.predict(X_test)
        
        # Calculate evaluation metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Mean Squared Error: {mse}")
        print(f"Mean Absolute Error: {mae}")
        print(f"R2 Score: {r2}")
        
        # Plot Actual vs Predicted
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, color='blue', alpha=0.3)
        plt.plot([0, max(y_test)], [0, max(y_test)], color='red', linestyle='--', linewidth=2)
        plt.xlabel('Actual Purchase Amount')
        plt.ylabel('Predicted Purchase Amount')
        plt.title(f'{name} Model: Actual vs Predicted')
        plt.show()

        # Save the model for future use
        joblib.dump(model, f"{name}_model.pkl")
        print(f"Model saved as {name}_model.pkl")

# Step 5: Main execution
if __name__ == "__main__":
    generate_data()  # Step 1: Generate Data
    df = pd.read_csv('customer_purchase_data.csv')  # Load data
    df = feature_engineering(df)  # Step 2: Feature Engineering
    
    # Step 3: Prepare Model
    X_train, X_test, y_train, y_test, models = prepare_model(df)
    
    # Step 4: Train and Evaluate
    train_and_evaluate(models, X_train, X_test, y_train, y_test)
