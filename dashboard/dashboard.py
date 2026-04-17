import os
import pandas as pd
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import joblib

# Load Data
data_path = 'outputs/customer_data.csv'
preds_path = 'outputs/churn_predictions.csv'
model_path = 'outputs/churn_model.pkl'

if not (os.path.exists(data_path) and os.path.exists(preds_path) and os.path.exists(model_path)):
    print("Required data files missing. Please run main.py first.")
    exit(1)

df_all = pd.read_csv(data_path)
df_preds = pd.read_csv(preds_path)

# Merge predictions back to main dataset (only test set will have risk segments, fill others if needed, or simply use df_preds for risk)
# To make it simple, we join. The test set has `risk_segment`.
df = pd.merge(df_all, df_preds[['customer_id', 'risk_segment', 'churn_probability', 'prediction']], on='customer_id', how='left')
df['risk_segment'] = df['risk_segment'].fillna('Not Evaluated (Train Set)')

# Load model for feature importance
rf_model = joblib.load(model_path)
feature_names = df_all.drop(columns=['customer_id', 'churn']).columns
# Use feature_names after encoding. In train_model.py, we did pd.get_dummies? No, we used LabelEncoder.
# So the columns used for training are exactly df_all.drop(columns=['customer_id', 'churn']).columns
importances = rf_model.feature_importances_
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

# Dash App
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = "Customer Churn Intelligence Platform"

# Layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H2("Customer Churn Intelligence Platform", className="text-center my-4"), width=12)
    ]),
    
    # KPI Row
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H5("Total Customers", className="card-title text-muted"),
                html.H3(id="kpi-total")
            ])
        ], className="shadow-sm"), width=3),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H5("Overall Churn Rate", className="card-title text-muted"),
                html.H3(id="kpi-churn-rate")
            ])
        ], className="shadow-sm"), width=3),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H5("Avg Monthly Charges", className="card-title text-muted"),
                html.H3(id="kpi-avg-charges")
            ])
        ], className="shadow-sm"), width=3),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H5("High Risk Customers", className="card-title text-muted"),
                html.H3(id="kpi-high-risk")
            ])
        ], className="shadow-sm"), width=3),
    ], className="mb-4"),
    
    # Filters Row
    dbc.Row([
        dbc.Col([
            html.Label("Contract Type"),
            dcc.Dropdown(
                id='filter-contract',
                options=[{'label': 'All', 'value': 'All'}] + [{'label': c, 'value': c} for c in df['contract_type'].unique()],
                value='All',
                clearable=False
            )
        ], width=6),
        dbc.Col([
            html.Label("Age Group"),
            dcc.Dropdown(
                id='filter-age',
                options=[{'label': 'All', 'value': 'All'}] + [{'label': c, 'value': c} for c in df['age_group'].unique()],
                value='All',
                clearable=False
            )
        ], width=6),
    ], className="mb-4"),
    
    # Charts Row 1
    dbc.Row([
        dbc.Col(dcc.Graph(id='chart-churn-pie'), width=4),
        dbc.Col(dcc.Graph(id='chart-churn-contract'), width=4),
        dbc.Col(dcc.Graph(id='chart-monthly-charges'), width=4),
    ], className="mb-4"),
    
    # Charts Row 2
    dbc.Row([
        dbc.Col(dcc.Graph(id='chart-risk-segment'), width=6),
        dbc.Col(dcc.Graph(id='chart-feature-importance'), width=6),
    ], className="mb-4")

], fluid=True)

# Callbacks
@app.callback(
    [Output('kpi-total', 'children'),
     Output('kpi-churn-rate', 'children'),
     Output('kpi-avg-charges', 'children'),
     Output('kpi-high-risk', 'children'),
     Output('chart-churn-pie', 'figure'),
     Output('chart-churn-contract', 'figure'),
     Output('chart-monthly-charges', 'figure'),
     Output('chart-risk-segment', 'figure'),
     Output('chart-feature-importance', 'figure')],
    [Input('filter-contract', 'value'),
     Input('filter-age', 'value')]
)
def update_dashboard(contract, age):
    filtered_df = df.copy()
    
    if contract != 'All':
        filtered_df = filtered_df[filtered_df['contract_type'] == contract]
    if age != 'All':
        filtered_df = filtered_df[filtered_df['age_group'] == age]
        
    # KPIs
    total_customers = len(filtered_df)
    
    if total_customers > 0:
        churn_rate = filtered_df['churn'].mean() * 100
        avg_charges = filtered_df['monthly_charges'].mean()
    else:
        churn_rate = 0
        avg_charges = 0
        
    high_risk = len(filtered_df[filtered_df['risk_segment'] == 'High'])
    
    kpi_tot = f"{total_customers:,}"
    kpi_churn = f"{churn_rate:.1f}%"
    kpi_avg = f"₹{avg_charges:.2f}"
    kpi_risk = f"{high_risk:,}"
    
    # Charts
    # 1. Churn pie chart
    churn_counts = filtered_df['churn'].value_counts().reset_index()
    churn_counts.columns = ['Churn', 'Count']
    churn_counts['Churn Status'] = churn_counts['Churn'].map({1: 'Churned', 0: 'Retained'})
    fig_pie = px.pie(churn_counts, values='Count', names='Churn Status', title='Churn Distribution',
                     color='Churn Status', color_discrete_map={'Retained': '#2ECC71', 'Churned': '#E74C3C'})
    
    # 2. Churn rate by contract
    if total_customers > 0:
        contract_churn = filtered_df.groupby('contract_type')['churn'].mean().reset_index()
        contract_churn['churn'] = contract_churn['churn'] * 100
        fig_contract = px.bar(contract_churn, x='contract_type', y='churn', title='Churn Rate by Contract (%)',
                              labels={'contract_type': 'Contract Type', 'churn': 'Churn Rate (%)'}, color='churn', 
                              color_continuous_scale='Reds')
    else:
        fig_contract = px.bar(title='Churn Rate by Contract (%)')
        
    # 3. Monthly charges box plot
    fig_charges = px.box(filtered_df, x='churn', y='monthly_charges', title='Monthly Charges by Churn Status',
                         labels={'churn': 'Churn (1=Yes, 0=No)', 'monthly_charges': 'Monthly Charges (₹)'},
                         color='churn', color_discrete_map={0: '#3498DB', 1: '#E74C3C'})
    
    # 4. Risk Segment breakdown
    # Filter only test set for risk segment visualization to avoid noise of "Not Evaluated"
    risk_df = filtered_df[filtered_df['risk_segment'] != 'Not Evaluated (Train Set)']
    if len(risk_df) > 0:
        risk_counts = risk_df.groupby(['contract_type', 'risk_segment']).size().reset_index(name='count')
        fig_risk = px.bar(risk_counts, x='contract_type', y='count', color='risk_segment', 
                          title='Risk Segment Breakdown (Test Set)', barmode='stack',
                          color_discrete_map={'Low': '#2ECC71', 'Medium': '#F1C40F', 'High': '#E74C3C'})
    else:
        fig_risk = px.bar(title='Risk Segment Breakdown (Test Set)')
        
    # 5. Feature Importance
    fig_feat = px.bar(importance_df, x='Importance', y='Feature', orientation='h', title='Feature Importance (RF Model)')
    fig_feat.update_layout(yaxis={'categoryorder': 'total ascending'})
    
    return kpi_tot, kpi_churn, kpi_avg, kpi_risk, fig_pie, fig_contract, fig_charges, fig_risk, fig_feat


if __name__ == '__main__':
    print("Dashboard running at http://127.0.0.1:8050")
    try:
        app.run(debug=True, port=8050)
    finally:
        print("✓ dashboard.py completed successfully.")
