import os
import pandas as pd
import numpy as np

def main():
    """
    Generates realistic synthetic customer data for a churn prediction model.
    Saves the output to 'outputs/customer_data.csv'.
    """
    try:
        np.random.seed(42)
        n_customers = 5000

        # customer_id
        customer_id = [f"CUST_{i:04d}" for i in range(1, n_customers + 1)]

        # tenure_months (skewed toward shorter tenures)
        tenure_months = np.random.exponential(scale=24, size=n_customers)
        tenure_months = np.clip(tenure_months, 1, 72).astype(int)

        # monthly_charges (normally distributed)
        monthly_charges = np.random.normal(loc=70, scale=20, size=n_customers)
        monthly_charges = np.clip(monthly_charges, 20, 120).round(2)

        # total_charges
        noise = np.random.normal(loc=0, scale=50, size=n_customers)
        total_charges = (monthly_charges * tenure_months + noise).clip(0, None).round(2)

        # num_products
        num_products = np.random.randint(1, 6, size=n_customers)

        # support_calls (Poisson lambda=3)
        support_calls = np.random.poisson(lam=3, size=n_customers)

        # contract_type
        contract_type = np.random.choice(
            ['Month-to-Month', 'One Year', 'Two Year'],
            size=n_customers,
            p=[0.60, 0.25, 0.15]
        )

        # payment_method
        payment_method = np.random.choice(
            ['Electronic Check', 'Credit Card', 'Bank Transfer', 'Mailed Check'],
            size=n_customers,
            p=[0.35, 0.30, 0.25, 0.10]
        )

        # age_group
        age_group = np.random.choice(
            ['Young Adult', 'Middle Age', 'Senior'],
            size=n_customers,
            p=[0.30, 0.45, 0.25]
        )

        # signup_month & cohort
        signup_month = np.random.randint(1, 13, size=n_customers)
        signup_year = np.random.randint(2021, 2024, size=n_customers)
        cohort = [f"{year}-{month:02d}" for year, month in zip(signup_year, signup_month)]

        # Churn probability based on logistic function
        # Base log-odds
        log_odds = -2.0 
        
        # higher support_calls -> higher churn
        log_odds += 0.8 * support_calls
        
        # Month-to-Month -> highest churn
        contract_effect = np.where(contract_type == 'Month-to-Month', 2.5, 
                                   np.where(contract_type == 'One Year', -1.0, -2.5))
        log_odds += contract_effect
        
        # lower tenure -> higher churn
        log_odds -= 0.1 * tenure_months
        
        # higher monthly_charges -> slightly higher churn
        log_odds += 0.02 * (monthly_charges - 70)

        # Add some random noise
        log_odds += np.random.normal(loc=0, scale=0.1, size=n_customers)

        # Logistic function to get probabilities
        prob_churn = 1 / (1 + np.exp(-log_odds))
        
        # Adjust intercept to get ~27% churn rate if needed, here we just sample
        churn = (np.random.rand(n_customers) < prob_churn).astype(int)

        # Compile into DataFrame
        df = pd.DataFrame({
            'customer_id': customer_id,
            'tenure_months': tenure_months,
            'monthly_charges': monthly_charges,
            'total_charges': total_charges,
            'num_products': num_products,
            'support_calls': support_calls,
            'contract_type': contract_type,
            'payment_method': payment_method,
            'age_group': age_group,
            'signup_month': signup_month,
            'cohort': cohort,
            'churn': churn
        })

        # Generate outputs directory if it doesn't exist
        os.makedirs('outputs', exist_ok=True)
        
        # Save to CSV
        output_path = 'outputs/customer_data.csv'
        df.to_csv(output_path, index=False)

        print(f"Data saved to {output_path}")
        print(f"Shape: {df.shape}")
        print(f"Overall Churn Rate: {df['churn'].mean():.2%}")
        print("✓ generate_data.py completed successfully.")

    except Exception as e:
        print(f"Error in generate_data.py: {e}")
        raise

if __name__ == "__main__":
    main()
