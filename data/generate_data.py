import os
import pandas as pd
import numpy as np

def main():
    """
    Downloads the IBM HR Employee Attrition dataset.
    Cleans and standardizes the columns mapping for our Employee Retention Analytics Pipeline.
    """
    try:
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context

        print("Downloading IBM HR Employee Attrition dataset...")
        url = "https://raw.githubusercontent.com/IBM/employee-attrition-aif360/master/data/emp_attrition.csv"
        df = pd.read_csv(url)
        
        # Standardize target variable to binary
        df['attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
        df.drop(columns=['Attrition'], inplace=True)
        
        # Optional: Standardize Employee ID
        df = df.rename(columns={'EmployeeNumber': 'employee_id'})
        
        # Determine Cohort (Hire Year) by backdating from 2024 based on YearsAtCompany
        current_year = 2024
        df['hire_year'] = current_year - df['YearsAtCompany']
        df['cohort'] = df['hire_year'].astype(str) + "-01"
        
        # Drop completely zero-variance or useless columns for ML
        if 'EmployeeCount' in df.columns:
            df.drop(columns=['EmployeeCount'], inplace=True)
        if 'StandardHours' in df.columns:
            df.drop(columns=['StandardHours'], inplace=True)
        if 'Over18' in df.columns:
            df.drop(columns=['Over18'], inplace=True)

        # Generate outputs directory if it doesn't exist
        os.makedirs('outputs', exist_ok=True)
        
        # Save to CSV
        output_path = 'outputs/employee_data.csv'
        df.to_csv(output_path, index=False)

        print(f"Data saved to {output_path}")
        print(f"Shape: {df.shape}")
        print(f"Overall Attrition Rate: {df['attrition'].mean():.2%}")
        print("✓ generate_data.py completed successfully.")

    except Exception as e:
        print(f"Error in generate_data.py: {e}")
        raise

if __name__ == "__main__":
    main()
