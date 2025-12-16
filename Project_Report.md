# Credit Risk Probability Model: Final Report

## 1. Executive Summary

This report details the development of a credit risk model for a Buy-Now-Pay-Later (BNPL) service that leverages alternative data. The primary objective was to build a robust, interpretable, and reproducible machine learning pipeline to predict the probability of default. The final solution consists of an end-to-end `scikit-learn` pipeline that handles all data processing, feature engineering, and modeling. The model is served via a REST API for real-time predictions and can be used for batch processing.

The project successfully balances the need for predictive accuracy with the regulatory and business requirements for interpretability, as outlined by frameworks like Basel II. A Logistic Regression model was chosen for its transparency, and an optional Weight of Evidence (WoE) transformer was included to support a traditional scorecard approach.

---

## 2. Business Understanding & Objectives

As outlined in the project's `README.md`, credit risk modeling in a regulated environment demands more than just predictive accuracy. The key business drivers for this project were:

- **Interpretability and Governance**: In line with Basel II principles, the model must be transparent and auditable. This means being able to explain which factors drive a risk score and having a well-documented, governable process.
- **Proxy Target Risk**: The project uses `FraudResult` as a proxy for default, as clean contractual default data is often unavailable in alternative-data settings. This introduces a risk that the proxy may not perfectly align with true credit distress, a factor that must be monitored.
- **Model Choice Trade-offs**: A simple, interpretable model like **Logistic Regression with WoE** was favored over a more complex model like Gradient Boosting. This choice prioritizes transparency, stability, and ease of governance, which are critical for regulatory compliance and operational deployment.

---

## 3. Exploratory Data Analysis (EDA) - Key Insights

An extensive EDA was conducted in the `notebooks/eda.ipynb` notebook. The analysis of the transaction data yielded several critical insights that directly informed the feature engineering and modeling strategy:

1.  **Highly Imbalanced Target**: The target variable, `FraudResult`, is extremely imbalanced, with only **~0.2%** of transactions being fraudulent. This necessitates the use of appropriate evaluation metrics (e.g., Precision-Recall AUC) and modeling techniques (e.g., class weighting or resampling) to avoid building a useless model.

2.  **Redundant & Outlier-Prone Features**: The `Amount` and `Value` columns are almost perfectly correlated (0.99) and contain a significant number of outliers. This suggests that one can be dropped to reduce complexity and that transformations (e.g., log-transform, capping) are needed to handle the extreme values.

3.  **`CustomerId` as the Key for Behavioral Features**: While `CustomerId` has high cardinality, it is the essential identifier for creating powerful behavioral features. Aggregating a customer's transaction history (e.g., total spending, average transaction value, frequency) is crucial for capturing patterns that predict risk.

4.  **Useless Features to Discard**: `CountryCode` and `CurrencyCode` have zero variance and provide no predictive value. High-cardinality identifiers like `TransactionId` and `BatchId` are also not useful as direct model inputs. These were marked for removal.

---

## 4. Data Processing & Feature Engineering

A robust and reproducible feature engineering pipeline was built in `src/data_processing.py` using `scikit-learn`'s `Pipeline` and `ColumnTransformer` objects. This ensures that the exact same transformations are applied during both training and prediction.

The pipeline consists of the following automated steps:

1.  **Datetime Feature Extraction** (`TransactionDatetimeFeatures`):
    - Extracts `hour`, `day`, `month`, `year`, `dayofweek`, and a `is_weekend` flag from the `TransactionStartTime` column.

2.  **Customer Aggregate Features** (`CustomerAggregateFeatures`):
    - Computes customer-level aggregations based on their transaction history. Features include:
        - **Transaction Amount Stats**: `sum`, `mean`, `std`, `min`, `max`, `count`.
        - **Transaction Value Stats**: `sum`, `mean`, `std`, `min`, `max`.
        - **Uniqueness Counts**: Number of unique providers, products, categories, and channels used by each customer.

3.  **Standard Preprocessing** (`ColumnTransformer`):
    - **Missing Value Imputation**: Fills missing numerical values with the `median` and categorical values with the `most_frequent` value.
    - **Categorical Encoding**: Applies `OneHotEncoder` to all categorical features.
    - **Numerical Scaling**: Applies `StandardScaler` to all numerical features.

4.  **Weight of Evidence (WoE) Encoding** (`WOEEncoder`):
    - An optional transformer that uses the `xverse` library to apply WoE transformation to categorical features. This is particularly useful for building interpretable scorecards.

---

## 5. Modeling & Prediction Workflow

The end-to-end workflow is designed for reproducibility and ease of deployment.

- **Training (`src/train.py`)**: The training script constructs a single, unified `scikit-learn` pipeline that chains the entire preprocessing workflow with a `LogisticRegression` classifier. This single pipeline object is then trained on the raw input data and saved to a file (`models/model.joblib`) using `joblib`. This approach guarantees that all preprocessing steps, including imputers, scalers, and encoders, are saved as part of the model artifact.

- **Batch Prediction (`src/predict.py`)**: The prediction script loads the saved pipeline object and calls `.predict_proba()` directly on new, raw data. Because the pipeline handles all feature engineering internally, no separate preprocessing code is needed, which eliminates a common source of training-serving skew.

- **API Deployment (`src/api/main.py`)**: A `FastAPI` application provides a real-time prediction endpoint (`/predict`). It loads the same pipeline object and uses it to score incoming requests, converting the raw JSON feature payload into a DataFrame and passing it to the pipeline for scoring. This ensures consistency between batch and real-time predictions.

---

## 6. Conclusion & Next Steps

This project successfully delivered an end-to-end credit risk modeling pipeline that is robust, reproducible, and aligned with business and regulatory requirements for interpretability. By encapsulating all logic within a `scikit-learn` pipeline, the solution minimizes training-serving skew and simplifies deployment.

**Potential Next Steps**:

- **Hyperparameter Tuning**: Conduct a systematic search for the optimal hyperparameters for the `LogisticRegression` model.
- **Advanced Feature Engineering**: Explore more complex behavioral features, such as time-based patterns (e.g., time since last transaction) or rolling window aggregates.
- **Challenger Models**: Implement a more complex model, such as Gradient Boosting (e.g., `XGBoost` or `LightGBM`), as a challenger to the current Logistic Regression model to quantify the trade-off between performance and interpretability.
- **Monitoring**: Implement a monitoring system to track model performance, feature drift, and the stability of the relationship between the proxy target and true business outcomes.
