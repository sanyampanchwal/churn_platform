import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import shap
import matplotlib.pyplot as plt
import joblib

def main():
    """
    Trains a Random Forest classifier to predict customer churn.
    Generates SHAP explainability plots and creates churn risk segments.
    """
    try:
        data_path = 'outputs/customer_data.csv'
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"{data_path} not found. Please run generate_data.py first.")
            
        df = pd.read_csv(data_path)
        
        # Identify categorical columns to encode
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Ensure customer_id is not treated as a feature to encode if it's there
        if 'customer_id' in categorical_cols:
            categorical_cols.remove('customer_id')
            
        # Store label encoders
        encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            
        # Optional: Save encoders if needed later
        joblib.dump(encoders, 'outputs/label_encoders.pkl')

        # Predictors and Target
        X = df.drop(columns=['customer_id', 'churn'])
        y = df['churn']
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, stratify=y, random_state=42
        )
        
        # Train Random Forest
        rf = RandomForestClassifier(
            n_estimators=200, 
            max_depth=10, 
            class_weight='balanced', 
            random_state=42
        )
        rf.fit(X_train, y_train)
        
        # Predictions
        y_pred = rf.predict(X_test)
        y_pred_proba = rf.predict_proba(X_test)[:, 1]
        
        # Metrics
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        acc = accuracy_score(y_test, y_pred)
        print(f"ROC-AUC Score: {roc_auc:.4f}")
        print(f"Accuracy Score: {acc:.4f}")
        
        if acc < 0.80:
            print("Warning: Accuracy is below 80%.")
            
        # SHAP Explainability
        explainer = shap.TreeExplainer(rf)
        
        # Calculate SHAP values for test set (we sample to not be too slow, or just use all X_test if small)
        # 1000 rows in X_test is fast enough
        shap_values = explainer.shap_values(X_test)
        
        # Note: shap.TreeExplainer returns a list of shap values for random forest classifier in shap <= 0.41, 
        # but in newer versions it returns an Explanation object or an array. 
        # For a binary classifier, it often returns a list [shap_values_class_0, shap_values_class_1] or an array of shape (n, f, 2).
        # We will use the SHAP values for the positive class (class 1, i.e., Churn)
        if isinstance(shap_values, list):
            shap_values_churn = shap_values[1]
        elif len(shap_values.shape) == 3:
            shap_values_churn = shap_values[:, :, 1]
        else:
            shap_values_churn = shap_values
            
        # Bar plot
        plt.figure()
        shap.summary_plot(shap_values_churn, X_test, plot_type="bar", show=False)
        plt.title('SHAP Feature Importance (Bar)')
        plt.tight_layout()
        plt.savefig('outputs/shap_summary.png')
        plt.close()
        
        # Beeswarm plot (using summary_plot without plot_type which defaults to beeswarm / dot)
        plt.figure()
        shap.summary_plot(shap_values_churn, X_test, show=False)
        plt.title('SHAP Feature Impact (Beeswarm)')
        plt.tight_layout()
        plt.savefig('outputs/shap_beeswarm.png')
        plt.close()

        # Churn Risk Segmentation
        # First, reconstruct test dataframe to keep customer_id
        test_indices = X_test.index
        df_test = df.loc[test_indices].copy()
        
        # If we need the original categoricals, we can decode them back, or just save encoded.
        # The dash app requires 'contract_type' and 'age_group'. So better to merge probas to original loaded data!
        # Re-read the CSV to get original textual data
        df_original = pd.read_csv(data_path)
        df_output = df_original.loc[test_indices].copy()
        
        df_output['prediction'] = y_pred
        df_output['churn_probability'] = y_pred_proba
        
        def assign_risk(pr):
            if pr < 0.30:
                return 'Low'
            elif pr <= 0.60:
                return 'Medium'
            else:
                return 'High'
                
        df_output['risk_segment'] = df_output['churn_probability'].apply(assign_risk)
        
        df_output.to_csv('outputs/churn_predictions.csv', index=False)
        print(f"Risk segmentation saved to outputs/churn_predictions.csv")

        # Save model
        joblib.dump(rf, 'outputs/churn_model.pkl')
        print("✓ Model training complete.")
        print("✓ train_model.py completed successfully.")

    except Exception as e:
        print(f"Error in train_model.py: {e}")
        raise

if __name__ == "__main__":
    main()
