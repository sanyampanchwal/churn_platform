import os
import pandas as pd
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.io as pio
import joblib

# Set all Plotly charts to use the beautiful dark template
pio.templates.default = "plotly_dark"

# Load Data
data_path = 'outputs/employee_data.csv'
preds_path = 'outputs/flight_risk.csv'
model_path = 'outputs/attrition_model.pkl'

if not (os.path.exists(data_path) and os.path.exists(preds_path) and os.path.exists(model_path)):
    print("Required data files missing. Please run main.py first.")
    exit(1)

df_all = pd.read_csv(data_path)
df_preds = pd.read_csv(preds_path)

# Merge predictions back to main dataset
df = pd.merge(df_all, df_preds[['employee_id', 'flight_risk_segment', 'attrition_probability', 'prediction']], on='employee_id', how='left')
df['flight_risk_segment'] = df['flight_risk_segment'].fillna('Not Evaluated (Train Set)')

# Load model for feature importance
rf_model = joblib.load(model_path)
feature_names = df_all.drop(columns=['employee_id', 'attrition', 'cohort', 'hire_year'], errors='ignore').columns

importances = rf_model.feature_importances_
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

# Dash App Configuration using beautiful DARKLY theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "Employee Retention Intelligence Platform"

# Custom styling definitions
card_style = {"backgroundColor": "#2c2c2c", "color": "#f8f9fa", "border": "1px solid #444"}

app.layout = dbc.Container([
    # Headings
    dbc.Row([
        dbc.Col([
            html.H1("Employee Retention Intelligence Platform", className="text-center mt-5 mb-3 text-info fw-bold"),
            html.P("This dashboard visualizes historical corporate attrition patterns alongside real-time Artificial Intelligence predictions. Use the dropdown filters below to slice segments and observe how HR statistics and AI risk assessments update instantly.", 
                   className="text-center text-light mb-5 fs-5")
        ], width=12)
    ]),
    
    # KPI Row
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("Total Employees", className="card-title text-muted text-uppercase fw-bold"),
                html.H2(id="kpi-total", className="text-info")
            ])
        ], style=card_style, className="shadow-lg"), width=3),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("Overall Resignation Rate", className="card-title text-muted text-uppercase fw-bold"),
                html.H2(id="kpi-attrition-rate", className="text-danger")
            ])
        ], style=card_style, className="shadow-lg"), width=3),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("Avg Monthly Income", className="card-title text-muted text-uppercase fw-bold"),
                html.H2(id="kpi-avg-income", className="text-success")
            ])
        ], style=card_style, className="shadow-lg"), width=3),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("High Flight Risk Talent", className="card-title text-muted text-uppercase fw-bold"),
                html.H2(id="kpi-high-risk", className="text-warning")
            ])
        ], style=card_style, className="shadow-lg"), width=3),
    ], className="mb-5"),
    
    # Filters Row
    dbc.Row([
        dbc.Col([
            html.Label("Filter by Corporate Department:", className="fw-bold text-light"),
            dcc.Dropdown(
                id='filter-dept',
                options=[{'label': 'All Departments', 'value': 'All'}] + [{'label': c, 'value': c} for c in df['Department'].unique() if pd.notnull(c)],
                value='All',
                clearable=False,
                style={"color": "#000"}  # Needs to be black to read it easily due to generic dropdown limitations
            )
        ], width=6),
        dbc.Col([
            html.Label("Filter by Job Role:", className="fw-bold text-light"),
            dcc.Dropdown(
                id='filter-role',
                options=[{'label': 'All Roles', 'value': 'All'}] + [{'label': c, 'value': c} for c in df['JobRole'].unique() if pd.notnull(c)],
                value='All',
                clearable=False,
                style={"color": "#000"} 
            )
        ], width=6),
    ], className="mb-5 bg-dark p-4 rounded border border-secondary shadow-lg"),
    
    # Historical Analysis Section
    dbc.Row([
        dbc.Col([
            html.H3("1. Historical Demographics & Payroll Trends", className="mt-4 border-bottom border-info pb-2 text-info"),
            html.P("Visualizing raw volume, occupational relationships, and the financial spread of employees who have historically stayed vs. resigned.", className="text-light mb-4")
        ], width=12)
    ]),

    # Charts Row 1
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='chart-attrition-pie'),
            dbc.Alert("Proportion of total employees retained vs resigned in this specific view. A quickly identifying health-check metric.", color="dark", className="mt-2 text-center text-light border-secondary")
        ], width=4),
        dbc.Col([
            dcc.Graph(id='chart-attrition-dept'),
            dbc.Alert("Identifies which Departments act as the leading drivers of high turnover and massive HR loss.", color="dark", className="mt-2 text-center text-light border-secondary")
        ], width=4),
        dbc.Col([
            dcc.Graph(id='chart-monthly-income'),
            dbc.Alert("Shows mathematically if lower monthly income tightly correlates with high employee resignation.", color="dark", className="mt-2 text-center text-light border-secondary")
        ], width=4),
    ], className="mb-5"),
    
    # ML Section
    dbc.Row([
        dbc.Col([
            html.H3("2. Artificial Intelligence Insights", className="mt-2 border-bottom border-warning pb-2 text-warning"),
            html.P("Surfaces the Random Forest classifier's logic. Segments the test group into 'Flight-Risk' employees for HR intervention, and ranks exactly which variables the AI uses to make its decisions via Game Theory (SHAP).", className="text-light mb-4")
        ], width=12)
    ]),

    # Charts Row 2
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='chart-risk-segment'),
            dbc.Alert("Breaks down current active employees by their estimated risk of quitting in the future.", color="dark", className="mt-2 text-center text-light border-secondary")
        ], width=6),
        dbc.Col([
            dcc.Graph(id='chart-feature-importance'),
            dbc.Alert("SHAP Transparency: Reveals which variables (Overtime, Income, Distance) currently impact the AI's internal scoring the heaviest.", color="dark", className="mt-2 text-center text-light border-secondary")
        ], width=6),
    ], className="mb-5"),
    
    # Footer
    dbc.Row([
        dbc.Col(html.P("© Employee Retention Intelligence Platform. Powered by Random Forest & SHAP Analytics.", className="text-center text-muted mt-5"), width=12)
    ])

], fluid=True, style={"backgroundColor": "#121212", "padding": "2rem"})

# Callbacks
@app.callback(
    [Output('kpi-total', 'children'),
     Output('kpi-attrition-rate', 'children'),
     Output('kpi-avg-income', 'children'),
     Output('kpi-high-risk', 'children'),
     Output('chart-attrition-pie', 'figure'),
     Output('chart-attrition-dept', 'figure'),
     Output('chart-monthly-income', 'figure'),
     Output('chart-risk-segment', 'figure'),
     Output('chart-feature-importance', 'figure')],
    [Input('filter-dept', 'value'),
     Input('filter-role', 'value')]
)
def update_dashboard(dept, role):
    filtered_df = df.copy()
    
    if dept != 'All':
        filtered_df = filtered_df[filtered_df['Department'] == dept]
    if role != 'All':
        filtered_df = filtered_df[filtered_df['JobRole'] == role]
        
    # KPIs
    total_emp = len(filtered_df)
    
    if total_emp > 0:
        attrition_rate = filtered_df['attrition'].mean() * 100
        avg_income = filtered_df['MonthlyIncome'].mean()
    else:
        attrition_rate = 0
        avg_income = 0
        
    high_risk = len(filtered_df[filtered_df['flight_risk_segment'] == 'High'])
    
    kpi_tot = f"{total_emp:,}"
    kpi_att = f"{attrition_rate:.1f}%"
    kpi_avg = f"${avg_income:,.0f}"
    kpi_risk = f"{high_risk:,}"
    
    # Charts
    # 1. Attrition pie chart
    att_counts = filtered_df['attrition'].value_counts().reset_index()
    att_counts.columns = ['Attrition', 'Count']
    att_counts['Status'] = att_counts['Attrition'].map({1: 'Resigned', 0: 'Retained'})
    fig_pie = px.pie(att_counts, values='Count', names='Status', title='Overall Retention Status',
                     color='Status', color_discrete_map={'Retained': '#00bc8c', 'Resigned': '#e74c3c'})
    fig_pie.update_layout(title_x=0.5, margin=dict(t=50, b=20, l=20, r=20))
    
    # 2. Attrition rate by department
    if total_emp > 0:
        dept_att = filtered_df.groupby('Department')['attrition'].mean().reset_index()
        dept_att['attrition'] = dept_att['attrition'] * 100
        fig_dept = px.bar(dept_att, x='Department', y='attrition', title='Turnover Rate by Department (%)',
                          labels={'Department': '', 'attrition': 'Turnover Rate (%)'}, color='attrition', 
                          color_continuous_scale='Reds')
        fig_dept.update_layout(title_x=0.5)
    else:
        fig_dept = px.bar(title='Turnover Rate by Department (%)')
        
    # 3. Monthly Income box plot
    fig_income = px.box(filtered_df, x='attrition', y='MonthlyIncome', title='Monthly Income vs Attrition Outcome',
                         labels={'attrition': 'Attrition (1=Yes)', 'MonthlyIncome': 'Monthly Income ($)'},
                         color='attrition', color_discrete_map={0: '#3498DB', 1: '#E74C3C'})
    fig_income.update_layout(title_x=0.5)
    
    # 4. Risk Segment breakdown
    risk_df = filtered_df[filtered_df['flight_risk_segment'] != 'Not Evaluated (Train Set)']
    if len(risk_df) > 0:
        risk_counts = risk_df.groupby(['Department', 'flight_risk_segment']).size().reset_index(name='count')
        fig_risk = px.bar(risk_counts, x='Department', y='count', color='flight_risk_segment', 
                          title='Flight Risk Segment Tracking (Test Set)', barmode='stack',
                          color_discrete_map={'Low': '#00bc8c', 'Medium': '#f39c12', 'High': '#e74c3c'})
        fig_risk.update_layout(title_x=0.5)
    else:
        fig_risk = px.bar(title='Flight Risk Segment Tracking (Test Set)')
        
    # 5. Feature Importance
    fig_feat = px.bar(importance_df.head(15), x='Importance', y='Feature', orientation='h', title='Top 15 Drivers of Employee Resignation (Random Forest)')
    fig_feat.update_layout(yaxis={'categoryorder': 'total ascending'}, title_x=0.5)
    
    return kpi_tot, kpi_att, kpi_avg, kpi_risk, fig_pie, fig_dept, fig_income, fig_risk, fig_feat

if __name__ == '__main__':
    print("Dashboard running at http://127.0.0.1:8050")
    try:
        app.run(debug=True, port=8050)
    finally:
        print("✓ dashboard.py completed successfully.")
