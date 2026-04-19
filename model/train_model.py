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
    Trains a Random Forest classifier to predict Employee Attrition.
    Generates SHAP explainability plots and creates flight risk segments.
    """
    try:
        data_path = 'outputs/employee_data.csv'
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"{data_path} not found. Please run generate_data.py first.")
            
        df = pd.read_csv(data_path)
        
        # Identify categorical columns to encode
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if 'employee_id' in categorical_cols:
            categorical_cols.remove('employee_id')
        if 'cohort' in categorical_cols:
            categorical_cols.remove('cohort')
            
        encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            
        joblib.dump(encoders, 'outputs/label_encoders.pkl')

        # Predictors and Target
        X = df.drop(columns=['employee_id', 'attrition', 'cohort', 'hire_year'], errors='ignore')
        y = df['attrition']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, stratify=y, random_state=42
        )
        
        rf = RandomForestClassifier(
            n_estimators=200, 
            max_depth=10, 
            class_weight='balanced', 
            random_state=42
        )
        rf.fit(X_train, y_train)
        
        y_pred = rf.predict(X_test)
        y_pred_proba = rf.predict_proba(X_test)[:, 1]
        
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        acc = accuracy_score(y_test, y_pred)
        print(f"ROC-AUC Score: {roc_auc:.4f}")
        print(f"Accuracy Score: {acc:.4f}")
        
        # SHAP Explainability
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_test)
        
        if isinstance(shap_values, list):
            shap_values_attrition = shap_values[1]
        elif len(shap_values.shape) == 3:
            shap_values_attrition = shap_values[:, :, 1]
        else:
            shap_values_attrition = shap_values
            
        plt.figure()
        shap.summary_plot(shap_values_attrition, X_test, plot_type="bar", show=False)
        plt.title('SHAP Feature Importance (Bar)')
        plt.tight_layout()
        plt.savefig('outputs/shap_summary.png')
        plt.close()
        
        plt.figure()
        shap.summary_plot(shap_values_attrition, X_test, show=False)
        plt.title('SHAP Feature Impact (Beeswarm)')
        plt.tight_layout()
        plt.savefig('outputs/shap_beeswarm.png')
        plt.close()

        # Flight Risk Segmentation
        test_indices = X_test.index
        df_original = pd.read_csv(data_path)
        df_output = df_original.loc[test_indices].copy()
        
        df_output['prediction'] = y_pred
        df_output['attrition_probability'] = y_pred_proba
        
        def assign_risk(pr):
            if pr < 0.30:
                return 'Low'
            elif pr <= 0.50:
                return 'Medium'
            else:
                return 'High'
                
        df_output['flight_risk_segment'] = df_output['attrition_probability'].apply(assign_risk)
        
        df_output.to_csv('outputs/flight_risk.csv', index=False)
        print(f"Risk segmentation saved to outputs/flight_risk.csv")

        joblib.dump(rf, 'outputs/attrition_model.pkl')
        print("✓ Model training complete.")

    except Exception as e:
        print(f"Error in train_model.py: {e}")
        raise

if __name__ == "__main__":
    main()
