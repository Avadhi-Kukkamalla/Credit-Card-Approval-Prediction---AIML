from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

model = joblib.load("credit_approval_model.pkl")
model_columns = joblib.load("model_columns.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    gender = request.form["gender"]
    own_car = request.form["own_car"]
    own_realty = request.form["own_realty"]
    children = int(request.form["children"])
    income = float(request.form["income"])
    income_type = request.form["income_type"]
    education = request.form["education"]
    family_status = request.form["family_status"]
    housing_type = request.form["housing_type"]
    occupation = request.form["occupation"]
    age = float(request.form["age"])
    years_employed = float(request.form["years_employed"])
    family_members = float(request.form["family_members"])
    work_phone = int(request.form["work_phone"])
    phone = int(request.form["phone"])
    email = int(request.form["email"])

    input_dict = {col: 0 for col in model_columns}

    input_dict["CODE_GENDER"] = 1 if gender == "M" else 0
    input_dict["FLAG_OWN_CAR"] = 1 if own_car == "Y" else 0
    input_dict["FLAG_OWN_REALTY"] = 1 if own_realty == "Y" else 0
    input_dict["CNT_CHILDREN"] = children
    input_dict["AMT_INCOME_TOTAL"] = income
    input_dict["FLAG_WORK_PHONE"] = work_phone
    input_dict["FLAG_PHONE"] = phone
    input_dict["FLAG_EMAIL"] = email
    input_dict["CNT_FAM_MEMBERS"] = family_members
    input_dict["AGE_YEARS"] = age
    input_dict["YEARS_EMPLOYED"] = years_employed
    input_dict["IS_UNEMPLOYED"] = 1 if years_employed == 0 else 0

    income_col = f"NAME_INCOME_TYPE_{income_type}"
    if income_col in input_dict:
        input_dict[income_col] = 1

    edu_col = f"NAME_EDUCATION_TYPE_{education}"
    if edu_col in input_dict:
        input_dict[edu_col] = 1

    family_col = f"NAME_FAMILY_STATUS_{family_status}"
    if family_col in input_dict:
        input_dict[family_col] = 1

    housing_col = f"NAME_HOUSING_TYPE_{housing_type}"
    if housing_col in input_dict:
        input_dict[housing_col] = 1

    occ_col = f"OCCUPATION_TYPE_{occupation}"
    if occ_col in input_dict:
        input_dict[occ_col] = 1

    input_df = pd.DataFrame([input_dict])[model_columns]

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    result = "REJECTED (High Risk)" if prediction == 1 else "APPROVED (Low Risk)"

    return render_template("index.html", prediction_text=f"Result: {result} (Risk Score: {probability:.2%})")


@app.route("/batch", methods=["GET"])
def batch_form():
    return render_template("batch.html")


@app.route("/batch_predict", methods=["POST"])
def batch_predict():
    file = request.files["csv_file"]
    batch_df = pd.read_csv(file)

    results = []
    for _, row in batch_df.iterrows():
        input_dict = {col: 0 for col in model_columns}

        input_dict["CODE_GENDER"] = 1 if row.get("gender", "M") == "M" else 0
        input_dict["FLAG_OWN_CAR"] = 1 if row.get("own_car", "N") == "Y" else 0
        input_dict["FLAG_OWN_REALTY"] = 1 if row.get("own_realty", "N") == "Y" else 0
        input_dict["CNT_CHILDREN"] = row.get("children", 0)
        input_dict["AMT_INCOME_TOTAL"] = row.get("income", 0)
        input_dict["FLAG_WORK_PHONE"] = row.get("work_phone", 0)
        input_dict["FLAG_PHONE"] = row.get("phone", 0)
        input_dict["FLAG_EMAIL"] = row.get("email", 0)
        input_dict["CNT_FAM_MEMBERS"] = row.get("family_members", 1)
        input_dict["AGE_YEARS"] = row.get("age", 30)
        input_dict["YEARS_EMPLOYED"] = row.get("years_employed", 0)
        input_dict["IS_UNEMPLOYED"] = 1 if row.get("years_employed", 0) == 0 else 0

        for field, prefix in [
            ("income_type", "NAME_INCOME_TYPE_"),
            ("education", "NAME_EDUCATION_TYPE_"),
            ("family_status", "NAME_FAMILY_STATUS_"),
            ("housing_type", "NAME_HOUSING_TYPE_"),
            ("occupation", "OCCUPATION_TYPE_"),
        ]:
            val = row.get(field, "")
            col = f"{prefix}{val}"
            if col in input_dict:
                input_dict[col] = 1

        input_df = pd.DataFrame([input_dict])[model_columns]
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        results.append({
            "applicant_id": row.get("applicant_id", "N/A"),
            "result": "REJECTED (High Risk)" if prediction == 1 else "APPROVED (Low Risk)",
            "risk_score": f"{probability:.2%}"
        })

    return render_template("batch_results.html", results=results)


if __name__ == "__main__":
    app.run(debug=True)