import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import warnings
from sklearn.exceptions import UndefinedMetricWarning

# countvectorizer - used to convert text into numbers based on how many times that text has appeared.
# train_test_split - divides into 2 parts: one for training and one for testing.
# multinomialNB - term frequency in a dataset.

# Suppress undefined metric warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# Read dataset from CSV
df = pd.read_csv("C:\\Users\\rajac\\Downloads\\text.csv")


# Extract the emails (text) and labels
emails = df['text'].values
labels = df['label'].values  # Corrected from 'valuessss' to 'values'

# Here, it will read all the emails and then check how many times there are unique words.
# fit part learns words and transform converts it into the matrix.
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(emails)

# Training 80% and testing 20%. 
x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=0.2, random_state=42)

# Train the model
model = MultinomialNB()
model.fit(x_train, y_train)

# Predict and evaluate
y_pred = model.predict(x_test)

# Here, the accuracy_score predicts the model that it got right.
# classification gives a report
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# User input for testing
input_user = input("Enter email:\n")
input_user_vectorized = vectorizer.transform([input_user])
predicted_label = model.predict(input_user_vectorized)

print("\nNew email prediction:")
if predicted_label[0] == 1:
    print("Spam")
else:
    print("Not spam")

# Print the evaluation metrics
print("Accuracy:", accuracy)
print("Classification:\n", report)

# scikit-learn for ML tools
# pip install scikit-learn
