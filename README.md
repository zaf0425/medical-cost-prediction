# Explainable Medical Cost Prediction System

Predicting individual medical insurance charges with machine learning, and explaining *why* each prediction is made. Six regression models are compared on a public insurance dataset, and the best one (LightGBM) is opened up with **SHAP** (global and local feature attribution) and **LIME** (local verification). The notebook also includes a Low/Medium/High risk classifier and a Gradio web app with plain-language explanations.

*Course project for CSE427: Machine Learning, BRAC University.*

## Motivation

High-accuracy models such as gradient-boosted trees are black boxes, which is a problem when the output is a financial estimate that insurers, providers and patients need to trust. This project aims to be both accurate and explainable, so a user can see **why** a given cost was predicted for them.

## Objectives

- Predict medical insurance charges accurately
- Compare linear, tree-based, boosted and neural models
- Give global explanations (which features matter overall) and local explanations (why this patient got this estimate)
- Cross-check SHAP explanations against LIME

## Dataset

- **Source:** public medical insurance cost dataset (`insurance.csv`)
- **Size:** 1,338 beneficiaries, 7 columns (age, sex, BMI, children, smoker, region, charges), no missing values, 1 duplicate row removed
- **Target:** `charges`, ranging from about $1,122 to $63,770 and strongly right-skewed (skewness 1.51)
- A second, larger dataset was explored but **deliberately excluded** from modeling: its columns (insurance coverage %, hospital admissions, medication count) are consequences of high cost, so using them would be target leakage.

## Approach

1. **EDA:** target distribution, outliers (kept, since they are real high-cost smokers), correlations, and region analysis (small effect on cost, so `region` was dropped).
2. **Key finding:** a strong **smoker × BMI interaction**. BMI is almost irrelevant for non-smokers (r ≈ 0.08) but strongly related to cost for smokers (r ≈ 0.80).
3. **Preprocessing and feature engineering**
   - `log1p` transform of the target (skewness 1.51 → −0.09); predictions are converted back with `expm1` and all metrics are reported in dollars
   - Binary encoding of sex and smoker
   - Engineered features: `smoker_x_bmi`, `age_squared`, `bmi_category` (WHO bins), `is_obese_smoker`
   - Stratified 80/20 split on smoker status, `StandardScaler` fit on training data only
4. **Models:** Linear Regression (baseline), Decision Tree, Random Forest, XGBoost, LightGBM, and an Artificial Neural Network (Keras)
5. **Explainability:** SHAP summary, dependence and waterfall plots; LIME on the highest- and lowest-cost test patients
6. **Risk tiers:** Low / Medium / High tiers defined from training-set tertiles, classified with a balanced Random Forest
7. **Application:** Gradio interface taking age, sex, BMI, smoker status and children, returning the predicted cost, risk tier and an explanation (optional natural-language summary via the Gemini API)

## Results

Test-set performance, evaluated in dollars:

| Model | MAE | RMSE | R² | MAPE |
|---|---|---|---|---|
| **LightGBM** | **$1,133** | $3,342 | **0.92** | **8.75%** |
| XGBoost | $1,298 | $3,522 | 0.91 | 8.73% |
| Random Forest | $1,736 | $3,461 | 0.92 | 19.15% |
| Decision Tree | $1,786 | $3,305 | 0.92 | 16.25% |
| ANN | $3,049 | $5,535 | 0.79 | 24.60% |
| Linear Regression | $3,570 | $7,544 | 0.61 | 23.76% |

LightGBM cuts MAE by about 68% compared with the linear baseline.

**Explainability findings**

- **Smoking status** is the most influential feature, followed by **age** and **BMI**.
- SHAP dependence plots show the risk is multiplicative: for smokers, the contribution of BMI rises sharply above about 30 kg/m², while for non-smokers it grows only slightly.
- For the highest-cost patient (about $52K), smoking and the smoker × BMI term are the strongest drivers. For the lowest-cost patient (about $1.1K), being a non-smoker and young pushes the estimate down. LIME agrees with SHAP on the main drivers.
- The Low/Medium/High risk classifier reaches about 93% accuracy on the test set.

## Repository Structure

```
.
├── Explainable_Medical_Cost_Prediction_System.ipynb   # EDA, preprocessing, models, SHAP/LIME, Gradio app
├── Project_Report.pdf                                 # Full paper-style report
└── PRESENTATION.pdf                                   # Course presentation slides
```

## Getting Started

The notebook was built in **Google Colab**. It looks for `insurance.csv` locally first and falls back to a public Google Drive copy.

```bash
pip install numpy pandas scipy matplotlib seaborn scikit-learn xgboost lightgbm \
            tensorflow shap lime gradio google-genai jupyter
jupyter notebook Explainable_Medical_Cost_Prediction_System.ipynb
```

- The Gradio interface is defined in the last section; uncomment `demo.launch()` to open it.
- The natural-language summary needs a Gemini API key (set as a Colab secret or environment variable). Without a key, the app still works and skips the summary.

## Limitations and Future Work

- Small dataset (1,338 rows) and limited features
- Adding clinical data (pre-existing conditions, medical history, genetic factors) and real-world records would likely improve accuracy
- Deploying the Gradio app as a hosted web service

## Authors

- Zafra Musarrat Islam
- Karib Fiyaz
- Injam Ul Haque

Department of Computer Science (and Engineering), BRAC University, Dhaka, Bangladesh.
