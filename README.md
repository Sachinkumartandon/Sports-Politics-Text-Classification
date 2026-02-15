# Sports vs Politics Text Classification

## Overview

This project implements a binary text classification system that classifies news articles into:

- Sports
- Politics

The system uses TF-IDF feature representation and compares three machine learning models.

---

## Models Used

### 1. Multinomial Naive Bayes
A probabilistic classifier based on Bayes' theorem. It assumes conditional independence between features and performs well for text classification tasks.

### 2. Logistic Regression
A linear classifier that models class probability using the sigmoid function. It learns a linear decision boundary in feature space.

### 3. Linear Support Vector Machine (SVM)
A margin-based classifier that finds the optimal separating hyperplane between classes. It is effective for high-dimensional sparse text data.

---

## Feature Representation

Text documents are converted into numerical vectors using:

- TF-IDF (Term Frequency - Inverse Document Frequency)
- Unigram features

---

## Project Structure

├── classifier.py
├── sports_politics.csv
├── B23CM1033_prob4.pdf
├── bbc/
└── README.md

---

## Requirements

Install required dependencies:


---

## How to Run

1. Ensure `sports_politics.csv` is in the same directory as `classifier.py`.

2. Run the classifier:

python classifier.py


The script will:
- Load the dataset
- Split into training and testing sets
- Train all three models
- Print accuracy results and classification metrics
