from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
import os
import plotly.express as px

# Flask setup
app = Flask(__name__, template_folder="../Frontend/templates", static_folder="../Frontend/static")

# ============================
# Model Prediction Code
# ============================

# Define paths
model_path = os.path.join(os.path.dirname(__file__), "model", "logistic_regression_attrition.pkl")
scaler_path = os.path.join(os.path.dirname(__file__), "model", "scaler.pkl")

# Load model and scaler
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Feature columns expected by the model
feature_columns = [
    'Age', 'DistanceFromHome', 'EnvironmentSatisfaction', 'JobInvolvement',
    'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 'StockOptionLevel',
    'TotalWorkingYears', 'WorkLifeBalance', 'YearsAtCompany',
    'YearsInCurrentRole', 'YearsWithCurrManager', 'OverTime_Yes',
    'JobRole_Human Resources', 'JobRole_Laboratory Technician',
    'JobRole_Manager', 'JobRole_Manufacturing Director',
    'JobRole_Research Director', 'JobRole_Research Scientist',
    'JobRole_Sales Executive', 'JobRole_Sales Representative',
    'MaritalStatus_Married', 'MaritalStatus_Single',
    'BusinessTravel_Travel_Frequently', 'BusinessTravel_Travel_Rarely',
    'EducationField_Life Sciences', 'EducationField_Marketing',
    'EducationField_Medical', 'EducationField_Other',
    'EducationField_Technical Degree'
]


def predict_attrition(
    Age, DistanceFromHome, EnvironmentSatisfaction, JobInvolvement, JobLevel,
    JobSatisfaction, MonthlyIncome, StockOptionLevel, TotalWorkingYears,
    WorkLifeBalance, YearsAtCompany, YearsInCurrentRole,
    YearsWithCurrManager, OverTime, JobRole, MaritalStatus,
    BusinessTravel, EducationField
):
    input_data = {col: 0 for col in feature_columns}

    # Fill numeric values
    input_data.update({
        'Age': Age,
        'DistanceFromHome': DistanceFromHome,
        'EnvironmentSatisfaction': EnvironmentSatisfaction,
        'JobInvolvement': JobInvolvement,
        'JobLevel': JobLevel,
        'JobSatisfaction': JobSatisfaction,
        'MonthlyIncome': MonthlyIncome,
        'StockOptionLevel': StockOptionLevel,
        'TotalWorkingYears': TotalWorkingYears,
        'WorkLifeBalance': WorkLifeBalance,
        'YearsAtCompany': YearsAtCompany,
        'YearsInCurrentRole': YearsInCurrentRole,
        'YearsWithCurrManager': YearsWithCurrManager,
    })

    # One-hot encodings
    if OverTime == "Yes":
        input_data['OverTime_Yes'] = 1

    jobrole_col = f"JobRole_{JobRole}"
    if jobrole_col in input_data:
        input_data[jobrole_col] = 1

    if MaritalStatus != "Divorced":
        marital_col = f"MaritalStatus_{MaritalStatus}"
        input_data[marital_col] = 1

    travel_col = f"BusinessTravel_{BusinessTravel}"
    if travel_col in input_data:
        input_data[travel_col] = 1

    edu_col = f"EducationField_{EducationField}"
    if edu_col in input_data:
        input_data[edu_col] = 1

    # Scale and predict
    df_input = pd.DataFrame([input_data])
    df_input = pd.DataFrame(scaler.transform(df_input), columns=df_input.columns)

    prediction = model.predict(df_input)[0]
    probability = model.predict_proba(df_input)[0][1]

    result = "Yes (Attrition Likely)" if prediction == 1 else "No (Attrition Unlikely)"
    return result, probability


# ============================
# Routes
# ============================

# Landing page
@app.route('/')
def home():
    return render_template("home.html")


# Prediction page
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template("index.html")
    try:
        form = request.form
        result, probability = predict_attrition(
            float(form['Age']), float(form['DistanceFromHome']),
            float(form['EnvironmentSatisfaction']), float(form['JobInvolvement']),
            float(form['JobLevel']), float(form['JobSatisfaction']),
            float(form['MonthlyIncome']), float(form['StockOptionLevel']),
            float(form['TotalWorkingYears']), float(form['WorkLifeBalance']),
            float(form['YearsAtCompany']), float(form['YearsInCurrentRole']),
            float(form['YearsWithCurrManager']), form['OverTime'],
            form['JobRole'], form['MaritalStatus'], form['BusinessTravel'],
            form['EducationField']
        )
        return render_template("result.html", prediction=result, probability=f"{probability:.2f}")
    except Exception as e:
        return render_template("result.html", prediction=f"Error: {str(e)}", probability="")


# Dashboard route
@app.route('/dashboard')
def dashboard():
    try:
        # Load dataset
        csv_path = os.path.join(os.path.dirname(__file__), "HR-Employee-Attrition.csv")
        df = pd.read_csv(csv_path)


        # --- Visualization 1: Overall Attrition ---
        fig1 = px.pie(df, names='Attrition', title='Overall Attrition Rate',
                      color_discrete_sequence=px.colors.sequential.RdBu)

        # --- Visualization 2: Attrition by Department ---
        dept_counts = df.groupby(['Department', 'Attrition']).size().reset_index(name='Count')
        fig2 = px.bar(dept_counts, x='Department', y='Count', color='Attrition',
                      barmode='group', title='Attrition by Department')

        # --- Visualization 3: Attrition by Gender ---
        gender_counts = df.groupby(['Gender', 'Attrition']).size().reset_index(name='Count')
        fig3 = px.bar(gender_counts, x='Gender', y='Count', color='Attrition',
                      barmode='group', title='Attrition by Gender')

        # --- Visualization 4: Attrition by Job Role ---
        jobrole_counts = df.groupby(['JobRole', 'Attrition']).size().reset_index(name='Count')
        fig4 = px.bar(jobrole_counts, x='Count', y='JobRole', color='Attrition',
                      orientation='h', title='Attrition by Job Role')

        # --- Visualization 5: Age vs Attrition ---
        fig5 = px.box(df, x='Attrition', y='Age', color='Attrition', title='Attrition vs Age')

        # --- Visualization 6: Monthly Income vs Attrition ---
        income_mean = df.groupby('Attrition')['MonthlyIncome'].mean().reset_index()
        fig6 = px.bar(income_mean, x='Attrition', y='MonthlyIncome', color='Attrition',title='Average Monthly Income by Attrition',text='MonthlyIncome')
        fig6.update_traces(texttemplate='%{text:.0f}', textposition='outside')
        
        
        # Convert to HTML
        graphs = [
            fig1.to_html(full_html=False),
            fig2.to_html(full_html=False),
            fig3.to_html(full_html=False),
            fig4.to_html(full_html=False),
            fig5.to_html(full_html=False),
            fig6.to_html(full_html=False)
        ]

        return render_template("dashboard.html", graphs=graphs)
    except Exception as e:
        return f"Error generating dashboard: {str(e)}"


# ============================
# Run App
# ============================

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5000)
