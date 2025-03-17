# AI Decision Support App
## Version: 1.33
## Deployed on: Render (e.g., https://ai-decision-support.onrender.com).
Last Updated: March 17, 2025

A Streamlit-based web application designed to simplify decision-making across Finance, Retail, and Healthcare use cases. Train machine learning models, visualize insights, and make real-world predictions with confidence scores.

## Overview
The AI Decision Support App leverages a Random Forest Classifier to predict outcomes based on user-uploaded or demo datasets. It supports three modes:
1. Finance: Predicts loan default risk to approve or deny loans.
2. Retail: Forecasts sales demand to guide inventory decisions.
3. Healthcare: Assesses patient risk to recommend care levels.

Key features include data visualization, feature importance analysis, LIME explanations, and actionable prediction messages tailored to real-world scenarios.

## Features
1. Multi-Mode Support:
Finance: Predicts "Default" (Yes/No) with loan approval recommendations.
Retail: Predicts "Demand" (High/Low) with stocking advice.
Healthcare: Predicts "Risk" (High/Low) with care prioritization.

2. Data Input: Use built-in demo datasets or upload your own CSV/PDF files.

3. Model Training: Random Forest Classifier with 100 trees and tuned parameters for realistic predictions.

4. Visualizations:
   
     a. Box plots of features vs. target.

     b. Feature importance bar charts.
 
     c. LIME explanations for individual predictions.

5. Predictions: Probability-based outcomes with confidence thresholds (e.g., >70% for confident decisions).

6. Export: Download datasets, predictions, and explanations as CSV or PNG.

7. Real-World Logic: Enhanced messages like "Approve loan? Yes with caution" or "Stock heavily" based on prediction confidence.

## Prerequisites
a. Python: 3.8+

b. Dependencies: Listed in requirements.txt

c. Git: For version control and deployment

d. Render Account: For hosting (optional)

## Installation
Clone the Repository:
```bash
git clone https://github.com/itswale/ai-decision-support.git
cd ai-decision-support
```

Set Up a Virtual Environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install Dependencies:
```bash
pip install -r requirements.txt
```

Run Locally:
```bash
streamlit run app.py
```

Open your browser to http://localhost:8501.

## Usage
Launch the App:
Run locally or visit the deployed URL (e.g., https://ai-decision-support.onrender.com).

## Sidebar Configuration:
1. Mode: Select Finance, Retail, or Healthcare.

2. Data Source: Choose a demo dataset or upload a CSV/PDF.

3. Features: Pick input variables (e.g., Income, CreditScore).

4. Target: Select the outcome to predict (e.g., Default, Demand, Risk).

5. Click "Train Model" to build the classifier.

## Main Interface:
1. Dataset Preview: View and paginate your data.

2. Visualizations: Explore feature-target relationships and importance.

3. Performance: Check accuracy, precision, and recall.

4. Prediction: Enter values and get actionable results (e.g., "Approve loan? Yes - Suggested amount: $6000.00").

## Example Inputs:
1. Finance: Income: 65000, CreditScore: 780, LoanAmount: 6000, Employed → "Approve loan? Yes".

2. Retail: DailySales: 800, FootTraffic: 250, PromoActive: Yes → "Predicted demand: High - Stock heavily".

3. Healthcare: Age: 80, BloodPressure: 160, Smoker: Yes → "Patient risk: High - Urgent care recommended".

## File Structure
```bash
ai-decision-support/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── loan_data_samples.csv       # Example dataset (optional)
```


## Deployment on Render
Push to GitHub:
Initialize Git and push your code:
```bash
git init
git add app.py requirements.txt README.md
git commit -m "Deploy AI Decision Support App v1.33"
git remote add origin https://github.com/itswale/ai-decision-support.git
git branch -M main
git push -u origin main
```

## Set Up Render:
Log in to Render.

New → Web Service → Connect your GitHub repo.

## Configure:
```bash
Name: ai-decision-support

Branch: main

Runtime: Python 3

Build Command: pip install -r requirements.txt

Start Command: streamlit run app.py --server.port $PORT --server.address 0.0.0.0

Instance Type: Free

Click "Create Web Service."

Access: Once deployed, visit the provided URL (e.g., https://ai-decision-support.onrender.com).
```

## Example Dataset
Here’s a snippet of loan_data.csv for Finance mode:
```bash
CustomerID,Income,CreditScore,LoanAmount,EmploymentStatus,Default
CUST001,45000.0,720,10000.0,Employed,No
CUST002,25000.0,580,15000.0,Unemployed,Yes
CUST003,60000.0,800,5000.0,Self-Employed,No
...
```

## Upload this file to test Finance predictions.

Technical Details
Model: RandomForestClassifier (n_estimators=100, min_samples_split=5, random_state=42).

Thresholds:
Finance: >70% "No" default = confident approval, 50-70% = cautious approval, <50% = denial.

Retail: >70% "High" demand = heavy stocking, 50-70% = moderate, <50% = light.

Healthcare: >70% "High" risk = urgent care, 50-70% = monitor, <50% = routine.

## Libraries:
Streamlit: UI framework.

Scikit-learn: Machine learning.

Pandas/Numpy: Data handling.

Plotly/Matplotlib/LIME: Visualizations.

PDFPlumber: PDF parsing.

## Limitations
1. Small Datasets: Predictions may be less reliable with <50 rows (confidence varies).

2. Free Tier Hosting: Render’s free plan sleeps after 15 minutes of inactivity, causing a brief delay on restart.

3. Binary Targets: Works best with 2-3 unique target values (e.g., Yes/No, High/Low).

## Contributing
Fork the repository.
```bash
Create a branch (git checkout -b feature/your-feature).

Commit changes (git commit -m "Add your feature").

Push (git push origin feature/your-feature).

Open a Pull Request.
```

## License
This project is open-source under the MIT License (LICENSE). Feel free to use, modify, and distribute.
Contact
Author: [itswale] (replace with your name or leave as is)

GitHub: itswale

Issues: Report bugs or suggest features in the Issues tab.

