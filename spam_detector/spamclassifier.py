import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import string
import re

df = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'text']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  
    text = text.translate(str.maketrans('', '', string.punctuation))  
    text = text.strip()
    return text

df['cleaned_text'] = df['text'].apply(clean_text)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['cleaned_text'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
train_pred = model.predict(X_train)

print("\n=== Evaluation Results ===")
print(f"Train Accuracy: {accuracy_score(y_train, train_pred) * 100:.2f}%")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")

print("=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

print("=== Confusion Matrix ===")
cm = confusion_matrix(y_test, y_pred)
print(f"TP: {cm[1][1]}, FN: {cm[1][0]}, FP: {cm[0][1]}, TN: {cm[0][0]}")

while True:
    user_input = input("\nEnter a message to classify (type 'exit' to quit):\n")
    if user_input.lower() == 'exit':
        break
    cleaned = clean_text(user_input)
    vectorized_input = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized_input)[0]
    print("Prediction:", "Spam" if prediction == 1 else "Ham")