import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report


# Load dataset
df = pd.read_csv("sports_politics.csv")

X = df["text"]
y = df["label"]

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- Feature Extraction ----------------
# Using only unigram TF-IDF (no bigrams, no stopword removal)
vectorizer = TfidfVectorizer(ngram_range=(1,1))

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ---------------- Model 1: Naive Bayes ----------------
nb = MultinomialNB()
nb.fit(X_train_vec, y_train)
nb_pred = nb.predict(X_test_vec)

# ---------------- Model 2: Logistic Regression ----------------
lr = LogisticRegression(max_iter=500)
lr.fit(X_train_vec, y_train)
lr_pred = lr.predict(X_test_vec)

# ---------------- Model 3: Linear SVM ----------------
svm = LinearSVC()
svm.fit(X_train_vec, y_train)
svm_pred = svm.predict(X_test_vec)

# ---------------- Results ----------------
print("Naive Bayes Accuracy:", round(accuracy_score(y_test, nb_pred), 4))
print("Logistic Regression Accuracy:", round(accuracy_score(y_test, lr_pred), 4))
print("SVM Accuracy:", round(accuracy_score(y_test, svm_pred), 4))

print("\nDetailed Classification Report (SVM):")
print(classification_report(y_test, svm_pred))
