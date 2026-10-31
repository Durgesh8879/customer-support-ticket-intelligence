# Customer Support Ticket Intelligence System

### Machine Learning | NLP | Text Classification | FastAPI | Deployment

An end-to-end Natural Language Processing (NLP) project that classifies customer support messages into predefined intents using machine learning. The project compares multiple text-classification models and provides a deployed web application for making predictions on new customer messages.

**Live Demo:** [Try the application](https://customer-support-ticket-intelligence-1.onrender.com)

**Backend API:** [Explore the API documentation](https://customer-support-ticket-intelligence.onrender.com/docs)

---

## Project Overview

Customer support teams receive large volumes of messages related to orders, invoices, accounts, payments, and other issues. Manually identifying the purpose of every message can be time-consuming.

This project explores how machine learning and NLP can automatically classify customer messages into appropriate intent categories.

I built a text-classification pipeline, compared different machine learning approaches, evaluated their performance, and deployed the trained model through a FastAPI application.

The goal is to demonstrate an end-to-end machine learning workflow, from dataset preparation and feature engineering to model evaluation and deployment.

## Problem Statement

Given a customer message, predict the intent behind that message so it can potentially be routed to the appropriate customer support workflow.

**Example:**

**Input:**

> Where can I find my invoice?

**Expected prediction:**
`check_invoice`

The prediction depends on the trained model and the wording of the input.

## Key Features

* Customer message classification using NLP.
* Text preprocessing and TF-IDF feature extraction.
* Comparison of Logistic Regression and Linear Support Vector Machine (SVM).
* Character-level TF-IDF feature engineering.
* Evaluation using a held-out test dataset.
* FastAPI backend for model inference.
* Web interface for submitting customer messages.
* Deployment of the frontend and backend.

## Dataset

The project uses the Bitext customer support LLM chatbot training dataset from Hugging Face.

| Property       |  Value |
| -------------- | -----: |
| Total records  | 26,872 |
| Features       |      5 |
| Unique intents |     27 |
| Categories     |     11 |
| Missing values |      0 |
| Duplicate rows |      0 |

The dataset contains customer-support-related instructions and their corresponding categories, intents, and responses.

The target variable for this project is the customer intent.

## Tech Stack

| Technology            | Purpose                             |
| --------------------- | ----------------------------------- |
| Python                | Core programming language           |
| Pandas                | Data manipulation and analysis      |
| NumPy                 | Numerical operations                |
| Scikit-learn          | Model training and evaluation       |
| TF-IDF                | Text feature extraction             |
| Logistic Regression   | Baseline text-classification model  |
| Linear SVM            | Text-classification model           |
| Jupyter Notebook      | Experimentation and analysis        |
| FastAPI               | Backend API                         |
| Pydantic              | Request validation                  |
| HTML, CSS, JavaScript | Web interface                       |
| Git and GitHub        | Version control and project hosting |
| Render                | Application deployment              |

## Machine Learning Workflow

### 1. Data Exploration

* Inspected the dataset's shape and columns.
* Checked for missing values and duplicate records.
* Examined the number of unique intents and categories.
* Investigated potentially confusing intent pairs.

### 2. Train-Test Split

Used `GroupShuffleSplit` with a test size of 20% and `random_state=42`, grouping by the `instruction` column.

| Dataset  | Records | Approximate proportion |
| -------- | ------: | ---------------------: |
| Training |  21,466 |                    80% |
| Testing  |   5,406 |                    20% |
| Total    |  26,872 |                   100% |

All 27 intents were represented in both the training and testing sets.

Grouping by instruction was intended to reduce leakage from repeated instructions across the train-test split.

### 3. Feature Engineering

Applied TF-IDF vectorization to convert text into numerical features that machine learning models can process.

Two approaches were evaluated:

* **Word-level TF-IDF:** Represents text using word-based features.
* **Character-level TF-IDF:** Represents text using character sequences, which can capture spelling patterns, word fragments, and variations in how messages are written.

Character-level features were particularly effective at distinguishing messages with similar wording.

### 4. Model Training

The following models were trained and compared:

1. Logistic Regression with word-level TF-IDF.
2. Linear SVM with word-level TF-IDF.
3. Linear SVM with character-level TF-IDF.

### 5. Model Evaluation

The models were evaluated on the held-out test set.

## Model Performance

| Model               | Feature extraction     | Test accuracy |
| ------------------- | ---------------------- | ------------: |
| Logistic Regression | Word-level TF-IDF      |        97.56% |
| Linear SVM          | Word-level TF-IDF      |        97.85% |
| Linear SVM          | Character-level TF-IDF |        99.87% |

The character-level SVM achieved the highest reported test accuracy and was selected for the deployed inference pipeline.

**Important evaluation note:** The high test accuracy should not be interpreted as proof of equivalent performance on completely new, real-world customer messages. Additional generalization testing identified cases where predictions did not match the intended intent. Performance on unseen messages remains an important area for further evaluation.

## Error Analysis

Error analysis helped identify intent pairs that were difficult to distinguish.

Examples included:

* `check_invoice` vs. `get_invoice`
* `create_account` vs. `delete_account`
* `track_order` vs. `change_order`
* `cancel_order` vs. `change_order`

These errors demonstrate that high overall accuracy does not necessarily mean every intent is classified reliably.

Further improvements could include intent-specific precision and recall analysis, confusion-matrix inspection, more diverse evaluation data, and targeted testing of ambiguous messages.

## Application Architecture

The application separates the frontend from the model-serving backend.

```text
              User
                |
                v
        Web Application
        HTML / CSS / JS
                |
                v
           FastAPI API
                |
                v
        Input Validation
             Pydantic
                |
                v
       Character TF-IDF
           Vectorizer
                |
                v
         Trained Linear SVM
                |
                v
       Predicted Customer Intent
                |
                v
       Prediction Displayed
          to the User
```

### Backend

The FastAPI backend loads the saved character-level TF-IDF vectorizer and trained SVM model.

The `/predict` endpoint accepts a customer message and returns the predicted intent.

**Example request:**

```http
POST /predict
Content-Type: application/json
```

```json
{
  "message": "Where can I find my invoice?"
}
```

**Example response structure:**

```json
{
  "customer_message": "Where can I find my invoice?",
  "predicted_intent": "check_invoice"
}
```

The predicted intent shown above is illustrative; actual predictions depend on the model output.

**Interactive API documentation:** [Open FastAPI Swagger UI](https://customer-support-ticket-intelligence.onrender.com/docs)

## Project Structure

The project includes a Jupyter Notebook for experimentation, a FastAPI backend, saved model artifacts, and frontend files.

A simplified representation is:

```text
customer-support-ticket-intelligence/
│
├── support_ai.ipynb
├── main.py
├── requirements.txt
├── index.html
├── script.js
├── style.css
├── saved_models/
│   ├── char_svm_model.pkl
│   └── char_tfidf_vectorizer.pkl
└── README.md
```

This illustrates the expected organization; the exact repository structure may differ.

## Run the Project Locally

### Prerequisites

* Python installed.
* Git installed.
* Access to the project repository.

### 1. Clone the repository

```bash
git clone https://github.com/Durgesh8879/customer-support-ticket-intelligence.git
```

### 2. Navigate into the project directory

```bash
cd customer-support-ticket-intelligence
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Start the backend

```bash
uvicorn main:app --reload
```

### 6. Open the API documentation

Visit:

http://127.0.0.1:8000/docs

**Note:** The saved model files must be present at the paths expected by the backend. The dependency versions, model artifacts, and any additional frontend-serving configuration must also match the current implementation.

## Future Improvements

Potential next steps include:

* Evaluate the model on a genuinely independent dataset of real customer messages.
* Investigate data leakage and near-duplicate text patterns.
* Report per-intent precision, recall, and F1-score.
* Build a more comprehensive confusion-matrix analysis.
* Improve handling of ambiguous and out-of-distribution messages.
* Add confidence-based abstention for uncertain predictions.
* Introduce automated tests for the API and prediction pipeline.
* Improve the user interface and error handling.
* Monitor model performance as new customer messages arrive.

## Key Learnings

This project provided practical experience with:

* Text preprocessing and TF-IDF feature engineering.
* Comparing linear machine learning models for NLP.
* Training and evaluating multiclass text classifiers.
* Investigating misclassifications and generalization issues.
* Saving and loading trained model artifacts.
* Building a prediction API with FastAPI.
* Connecting a frontend to a machine learning backend.
* Deploying a machine learning application.

## Author

**Data Science Portfolio Project**

GitHub Repository: [Customer Support Ticket Intelligence](https://github.com/Durgesh8879/customer-support-ticket-intelligence)

Live Application: [Try the project](https://customer-support-ticket-intelligence-1.onrender.com)

---

*This project was developed for learning and portfolio demonstration. Predictions should be evaluated carefully before use in production customer support workflows.*
