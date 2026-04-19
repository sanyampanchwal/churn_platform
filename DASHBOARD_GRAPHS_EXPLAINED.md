# Visual Analytics Guide: Understanding Your Graphs

This comprehensive guide explains exactly what every visualization in the **Customer Churn Intelligence Platform** represents, how it is calculated, and what underlying data it relies on.

---

## Part 1: Interactive Dashboard Charts (Plotly Dash)

These charts are dynamically generated in real-time when running `dashboard.py`. They respond instantly to the filters you select (Contract Type & Age Group).

### 1. Churn Distribution (Pie Chart)
- **What it represents:** The raw, high-level proportion of users who have historically "Retained" vs. "Churned" within your current filtered view.
- **What it works on:** It counts the binary `churn` column (`1` = Churned, `0` = Retained) from the `outputs/customer_data.csv` dataset.
- **Business Use Case:** A quick health check of the business. If you filter to "Senior" aged users and see an abnormally large red pie slice, it signals a systemic demographic problem.

### 2. Churn Rate by Contract (Vertical Bar Chart)
- **What it represents:** Compares the likelihood of churning across different subscription commitments (e.g., "Month-to-Month" vs "One Year"). 
- **What it works on:** It groups the data by the `contract_type` column and calculates the mean mathematical average of the `churn` column (where churning is represented as 100%).
- **Business Use Case:** Normally, you will see "Month-to-Month" dominate this chart. It mathematically justifies business initiatives promoting 12-month lock-in periods at discounted rates.

### 3. Monthly Charges by Churn Status (Box Plot)
- **What it represents:** Shows the statistical distribution (median, quartiles, and outliers) of how much money users were paying per month, split definitively by whether they stayed or left.
- **What it works on:** Uses the `monthly_charges` column plotted on the Y-Axis natively against the binary `churn` column on the X-Axis. 
- **Business Use Case:** Proves whether pricing is the trigger for abandonment. If the "Churned" box is sitting significantly higher on the graph than the "Retained" box, it means expensive plans are directly forcing customers away.

### 4. Risk Segment Breakdown (Stacked Bar Chart)
- **What it represents:** Groups current, active customers into highly actionable buckets based on the Artificial Intelligence's prediction of their *future* flight risk.
- **What it works on:** It utilizes the Machine Learning classifications (`Low`, `Medium`, `High` risk) calculated in `churn_predictions.csv` and stacks them horizontally across the different `contract_type` values.
- **Business Use Case:** Used aggressively by Marketing and Customer Success teams. The red "High Risk" blocks represent specific real users who are currently active but on the immediate verge of canceling. These are the prime targets for immediate promotional interventions.

### 5. Feature Importance (Horizontal Bar Chart)
- **What it represents:** Absolute transparency into the "Black Box" of the Random Forest Artificial Intelligence. It shows exactly which customer traits the AI is looking at to make its decision.
- **What it works on:** This is an implementation of **SHAP (SHapley Additive exPlanations)**. It intercepts the mathematically trained Tree nodes from your saved `churn_model.pkl` weights and measures which columns statistically swung the model's opinion the heaviest.
- **Business Use Case:** Answers the CEO's fundamental question: *"Why is the AI predicting they will leave?"* If "Tech Support" or "Internet Service" shoot to the top of this graph, it means terrible tech support is the core reason the company is bleeding money.

---

## Part 2: Static Exploratory Data Analysis (EDA) Charts

These visuals are generated automatically by `analysis.py` during your pipeline build phase and saved directly into the `outputs/` folder.

### 6. Cohort Heatmap (`cohort_heatmap.png`)
- **What it represents:** A grid showing churn rates over time based on the exact month/year a block of users originally signed up.
- **What it works on:** Uses a derived `cohort` date column and groups them against `tenure_months` "buckets".
- **Business Use Case:** Spots historical product failures. For example, if users who joined in June 2023 have massive churn rates compared to September 2023, you can deduce that whatever promotional campaign or software update ran in June was heavily flawed.

### 7. Correlation Matrix (`correlation_matrix.png`)
- **What it represents:** A massive grid showing how strongly *every single variable* interacts with *every other variable*. Scale goes from -1 (Strong Negative relationship) to 1 (Strong Positive relationship).
- **What it works on:** Takes every raw column, numericizes categorical text using `pd.get_dummies()`, and runs a Pearson Correlation calculation across the entire dataframe.
- **Business Use Case:** Uncovers hidden mathematical secrets. It might aggressively prove that `PaymentMethod_ElectronicCheck` is heavily correlated with leaving, prompting action from the billing department.

### 8. Kaplan-Meier Survival Curves (`survival_curves.png`)
- **What it represents:** A time-degradation line graph. Instead of just answering "Did they churn?", it answers "At what exact month are they most likely to churn?"
- **What it works on:** Employs the `lifelines` statistical library. It plots `tenure_months` (time) against the occurrence of `churn` (the "death" event), segmented by `contract_type`.
- **Business Use Case:** Allows business forecasting to anticipate massive drop-offs. If the survival curve violently dips exactly at Month 12, the company learns it must aggressively start retention calls around Month 11.
