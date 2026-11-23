# Predictive Modeling and Risk Scoring for Bank Customer Churn

A machine learning project that predicts which bank customers are likely to leave, gives each customer a churn probability and a Low / Medium / High risk band, explains the main churn drivers, and serves everything through an interactive Streamlit dashboard.

Built as part of the Data Analyst Internship at Unified Mentor.

## Live links

| Item | Link |
|---|---|
| Streamlit app | [add deployed link] |
| Research paper | [add link] |
| Project explanation video | [add link] |
| Internship experience and feedback video | [add link] |

## Why this project

Customer churn reduces customer lifetime value, destabilises revenue and weakens cross-sell potential. Traditional analysis explains churn after it happens. This project predicts it **before** the customer leaves, so a bank can run targeted retention instead of broad, costly campaigns.

## Dataset

| Item | Detail |
|---|---|
| Source | European Bank customer data (Unified Mentor project dataset) |
| Size | 10,000 customers, 14 columns |
| Target | `Exited` (1 = churned, 0 = retained) |
| Churn rate | 20.4% |
| Data quality | No missing values, no duplicate customer IDs |

Columns include credit score, geography (France, Spain, Germany), gender, age, tenure, balance, number of products, credit card ownership, activity status and estimated salary. `CustomerId`, `Surname` and the constant `Year` column are dropped.

## Methodology

| Step | What was done |
|---|---|
| Preprocessing | Dropped identifiers, one-hot encoded Geography and Gender, standardised numeric features inside a scikit-learn Pipeline (fitted on training data only to avoid leakage) |
| Feature engineering | Balance-to-salary ratio, product density, engagement-product interaction, age-tenure interaction, zero-balance flag |
| Split and validation | Stratified 80/20 train-test split, 5-fold stratified cross-validation on the training set |
| Models | Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost |
| Model selection | Highest cross-validated ROC-AUC (test set used once, for reporting) |
| Threshold | Tuned on out-of-fold training predictions to maximise F1 |
| Explainability | Permutation importance, partial dependence plots, SHAP |

## Results

| Model | CV ROC-AUC | Test ROC-AUC |
|---|---|---|
| **Gradient Boosting (selected)** | **0.864** | **0.872** |
| XGBoost | 0.861 | 0.866 |
| Random Forest | 0.858 | 0.867 |
| Decision Tree | 0.827 | 0.838 |
| Logistic Regression (baseline) | 0.766 | 0.776 |

With the tuned threshold of 0.27, the selected model reaches recall 0.68, precision 0.59 and F1 0.63 on the test set.

| Risk band | Probability | Actual churn rate (test set) |
|---|---|---|
| Low | below 0.30 | 8.9% |
| Medium | 0.30 to 0.60 | 43.7% |
| High | 0.60 and above | 87.4% |

Contacting the riskiest 20% of customers reaches about 63% of all churners.

### Key findings

- **Age** and **number of products** are the strongest churn drivers, followed by geography, balance and activity status.
- Customers with **two products** churn least (7.6%). Customers with one product churn more, and those with three or four churn very often (small groups, so treat with caution).
- **Inactive** customers churn almost twice as often as active ones.
- Churn is highest for customers aged **50 to 59**, and **Germany** has roughly double the churn of France and Spain.
- Credit card ownership, tenure, credit score and salary have little relationship with churn.

Numbers come from a run with `random_state = 42`. Small differences are possible with other library versions.

## Streamlit dashboard

| Tab | What it does |
|---|---|
| Risk calculator | Enter a customer profile and get the churn probability, risk band, churn flag and suggested action |
| Probability distribution | Shows where the customer sits among all customers, and actual churn per risk band |
| Feature importance | Permutation importance chart, plus SHAP and partial dependence plots |
| What-if simulator | Change engagement and product values and see how the probability moves, with a one-variable sweep chart |
| Model and method | Performance metrics, model comparison, method notes and limitations |

## Repository structure

```
.
├── README.md
├── Bank_Churn_Prediction.ipynb   # EDA, modeling, evaluation, explainability
├── churn_model.py                # Training script (creates the artifacts folder)
├── App/
│   ├── streamlit_app.py          # Streamlit dashboard
│   ├── requirements.txt          # Dependencies for the app
│   └── artifacts/                # Saved model, metadata, scores and plots
├── Streamlit_App.ipynb           # Notebook that writes and launches the app
└── European_Bank_-_European_Bank__1_.csv
```

Adjust the folder names above to match your repository layout. The `artifacts/` folder must sit next to `streamlit_app.py`.

## How to run

**1. Install dependencies**

```
pip install -r requirements.txt
```

**2. Train the model (creates the `artifacts/` folder)**

```
python churn_model.py --data European_Bank_-_European_Bank__1_.csv
```

Or run `Bank_Churn_Prediction.ipynb` from top to bottom.

**3. Start the dashboard**

```
streamlit run streamlit_app.py
```

Then open http://localhost:8501.

**Note:** the saved model is a scikit-learn pickle. Use the same scikit-learn version for training and for the app (this project uses 1.7.2).

## Limitations and responsible use

- The data is a single snapshot with no time dimension, so the model needs monitoring and periodic retraining.
- Scores show association, not cause. Use them to prioritise outreach, not to make adverse decisions about customers.
- Gender and geography are model inputs. Check for fair treatment across groups before using scores to target offers.
- Estimates for rare groups (for example customers with three or four products) are less reliable.
- The decision threshold and risk bands depend on assumed retention costs and should be recalibrated with real figures.

## Tech stack

Python, pandas, NumPy, scikit-learn, XGBoost, SHAP, Matplotlib, Seaborn, Plotly, Streamlit, joblib.

## Author

Naina, Master's in FinTech (MMSFT). Data Analyst Internship, Unified Mentor.
