# 1. Import necessary libraries
# !pip install scikit-learn pandas

# 2. Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 3. Load and prepare the dataset
# You'll need a dataset of emails labeled as spam or not spam (ham).
# You can find public datasets online or create your own.
# Load the dataset into a Pandas DataFrame.
# Assuming your dataset has columns 'text' (email content) and 'label' (spam/ham):

df = pd.read_csv("spam.csv", encoding='latin-1')
# print(df.head())

# 4. Split the dataset
x = df['text']
y = df['label']
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)  # Adjust test_size as needed

# test_size y el random_state

# 5. Feature extraction
# Convert text emails into numerical features using TF-IDF:
vectorizer = TfidfVectorizer()
x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.transform(x_test)

# 6. Train the model
# Use a Logistic Regression model (or any other suitable classification model):
model = LogisticRegression()
model.fit(x_train_vec, y_train)

# 7. Evaluate the model
y_pred = model.predict(x_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# 8. Use the model for prediction
def predict_spam(email_text):
    email_vec = vectorizer.transform([email_text])
    prediction = model.predict(email_vec)[0]
    # Assuming 1 represents spam, 0 represents ham
    return "Spam" if prediction == 1 else "Ham"

# Example usage
# email = "This is an example email."
# email = "Congratulations! You've won a free iPhone. Click here to claim now!"
# email = "Hi Jose, Please send me the new project details"
email = input("Enter an email: ")

result = predict_spam(email)
print(f"The email is classified as: {result}")
