import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter

def main():
    """
    Performs Exploratory Data Analysis (EDA) on the HR datset.
    Generates and saves cohort analysis, correlation matrix, survival curves, and distribution plots.
    """
    try:
        data_path = 'outputs/employee_data.csv'
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"{data_path} not found. Please run generate_data.py first.")
            
        df = pd.read_csv(data_path)
        
        # A) COHORT ANALYSIS
        # Group by cohort and YearsAtCompany buckets to calculate attrition rate
        df['tenure_bucket'] = pd.cut(df['YearsAtCompany'], bins=[-1, 2, 5, 10, 20, 40], 
                                     labels=['0-2', '3-5', '6-10', '11-20', '21+'])
        
        cohort_data = df.groupby(['cohort', 'tenure_bucket'], observed=True)['attrition'].mean().unstack()
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(cohort_data, annot=True, fmt=".2f", cmap="YlOrRd")
        plt.title('Cohort Analysis: Attrition Rate by Hire Year and Tenure')
        plt.ylabel('Cohort (Hire Year)')
        plt.xlabel('Tenure Bucket (Years)')
        plt.tight_layout()
        plt.savefig('outputs/cohort_heatmap.png')
        plt.close()

        # B) CORRELATION MATRIX
        # Drop IDs/cohorts
        df_corr = df.drop(columns=['employee_id', 'cohort', 'tenure_bucket', 'hire_year'], errors='ignore')
        df_corr = pd.get_dummies(df_corr, drop_first=True)
        
        corr_matrix = df_corr.corr()
        
        plt.figure(figsize=(24, 20))
        sns.heatmap(corr_matrix, cmap="coolwarm", center=0)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig('outputs/correlation_matrix.png')
        plt.close()

        # C) SURVIVAL ANALYSIS (Kaplan-Meier)
        plt.figure(figsize=(10, 6))
        kmf = KaplanMeierFitter()
        
        for dept in df['Department'].unique():
            mask = df['Department'] == dept
            kmf.fit(df[mask]['YearsAtCompany'], event_observed=df[mask]['attrition'], label=dept)
            kmf.plot_survival_function()
            
        plt.title('Employee Retention Survival Analysis by Department')
        plt.xlabel('Years At Company')
        plt.ylabel('Retention Probability')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('outputs/survival_curves.png')
        plt.close()

        # D) DISTRIBUTION PLOTS
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        sns.barplot(data=df, x='Department', y='attrition', ax=axes[0], palette='viridis')
        axes[0].set_title('Attrition Rate by Department')
        axes[0].tick_params(axis='x', rotation=45)
        
        sns.barplot(data=df, x='JobRole', y='attrition', ax=axes[1], palette='viridis')
        axes[1].set_title('Attrition Rate by Job Role')
        axes[1].tick_params(axis='x', rotation=80)
        
        sns.barplot(data=df, x='MaritalStatus', y='attrition', ax=axes[2], palette='viridis')
        axes[2].set_title('Attrition Rate by Marital Status')
        
        plt.tight_layout()
        plt.savefig('outputs/attrition_by_category.png')
        plt.close()

        print("EDA complete. 4 plots saved to outputs/")
        print("✓ analysis.py completed successfully.")

    except Exception as e:
        print(f"Error in analysis.py: {e}")
        raise

if __name__ == "__main__":
    main()
