import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter

def main():
    """
    Performs Exploratory Data Analysis (EDA) on the customer data.
    Generates and saves cohort analysis, correlation matrix, survival curves, and distribution plots.
    """
    try:
        data_path = 'outputs/customer_data.csv'
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"{data_path} not found. Please run generate_data.py first.")
            
        df = pd.read_csv(data_path)
        
        # A) COHORT ANALYSIS
        # Group by cohort and tenure buckets to calculate churn rate
        # We can use pd.qcut or pd.cut to bucket tenures
        df['tenure_bucket'] = pd.cut(df['tenure_months'], bins=[0, 12, 24, 36, 48, 60, 72], 
                                     labels=['0-12', '13-24', '25-36', '37-48', '49-60', '61-72'])
        
        cohort_data = df.groupby(['cohort', 'tenure_bucket'])['churn'].mean().unstack()
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(cohort_data, annot=True, fmt=".2f", cmap="YlGnBu")
        plt.title('Cohort Analysis: Churn Rate by Cohort and Tenure')
        plt.ylabel('Cohort (Signup Year-Month)')
        plt.xlabel('Tenure Bucket (Months)')
        plt.tight_layout()
        plt.savefig('outputs/cohort_heatmap.png')
        plt.close()

        # B) CHURN CORRELATION MATRIX
        # Drop columns like customer_id, dates etc before encoding
        df_corr = df.drop(columns=['customer_id', 'cohort', 'tenure_bucket'])
        df_corr = pd.get_dummies(df_corr, drop_first=True)
        
        corr_matrix = df_corr.corr()
        
        plt.figure(figsize=(16, 12))
        sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", center=0)
        plt.title('Churn Correlation Matrix')
        plt.tight_layout()
        plt.savefig('outputs/correlation_matrix.png')
        plt.close()

        # C) SURVIVAL ANALYSIS (Kaplan-Meier)
        plt.figure(figsize=(10, 6))
        kmf = KaplanMeierFitter()
        
        for contract in df['contract_type'].unique():
            mask = df['contract_type'] == contract
            # 'tenure_months' is the duration, 'churn' is the event observed
            kmf.fit(df[mask]['tenure_months'], event_observed=df[mask]['churn'], label=contract)
            kmf.plot_survival_function()
            
        plt.title('Survival Analysis by Contract Type')
        plt.xlabel('Tenure (Months)')
        plt.ylabel('Survival Probability')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('outputs/survival_curves.png')
        plt.close()

        # D) DISTRIBUTION PLOTS
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        sns.barplot(data=df, x='contract_type', y='churn', ax=axes[0])
        axes[0].set_title('Churn Rate by Contract Type')
        axes[0].tick_params(axis='x', rotation=45)
        
        sns.barplot(data=df, x='age_group', y='churn', ax=axes[1])
        axes[1].set_title('Churn Rate by Age Group')
        axes[1].tick_params(axis='x', rotation=45)
        
        sns.barplot(data=df, x='payment_method', y='churn', ax=axes[2])
        axes[2].set_title('Churn Rate by Payment Method')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('outputs/churn_by_category.png')
        plt.close()

        print("EDA complete. 4 plots saved to outputs/")
        print("✓ analysis.py completed successfully.")

    except Exception as e:
        print(f"Error in analysis.py: {e}")
        raise

if __name__ == "__main__":
    main()
