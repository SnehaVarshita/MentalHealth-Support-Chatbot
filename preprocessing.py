import os
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Download necessary nltk data files
nltk.download('punkt')
nltk.download('wordnet')

# Load the dataset
file_path = './combined_mental_health_dataset.json'
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

with open(file_path, 'r') as f:
    data = json.load(f)

# Check if the dataset has intents
if 'intents' not in data or not data['intents']:
    raise ValueError("No intents found in the dataset.")

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Helper function to tokenize and lemmatize
def process_text(text):
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(lemmatized_tokens)

# Prepare data for saving and processing
processed_data = []
texts = []
tags = []

# Debug: Check the structure of the dataset
print("Dataset structure:", json.dumps(data, indent=4))

for intent in data['intents']:
    print(f"Processing intent: {intent['tag']}")
    for pattern in intent['patterns']:
        if pattern.strip():  # Skip empty patterns
            processed_pattern = process_text(pattern)
            texts.append(processed_pattern)
            tags.append(intent['tag'])
            # Check if responses exist, if not set to empty list
            responses = intent.get('responses', [])  # Default to empty list if no responses
            processed_data.append({
                'tag': intent['tag'],
                'patterns': processed_pattern,
                'responses': responses  # Include responses
            })

# Check for consistent lengths
assert len(texts) == len(tags), "Inconsistent lengths between texts and tags"

# Diagnostic print to check if texts are processed correctly
print("Sample processed texts (first 5):", texts[:5])

# Save processed data
processed_file_path = './tprocessed_mental_health_data.json'
with open(processed_file_path, 'w') as f:
    json.dump({'intents': processed_data}, f, indent=4)
print(f"Processed data saved to {processed_file_path}")

# Convert texts into numerical features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
y = tags
# Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a classifier (Logistic Regression in this case)
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Calculate accuracy on the test set
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy on the test set: {accuracy * 100:.2f}%")

# Example of how to use the model on a new input
new_input = "I feel sad and lonely"
processed_input = vectorizer.transform([process_text(new_input)])
predicted_tag = classifier.predict(processed_input)[0]
print(f"Predicted tag for input '{new_input}': {predicted_tag}")

# View the training and testing datasets (first 5 entries)
print("\nTraining Data (first 5 entries):")
print(pd.DataFrame(X_train.toarray(), columns=vectorizer.get_feature_names_out()).head())

print("\nTraining Labels (first 5 entries):")
print(y_train[:5])

print("\nTesting Data (first 5 entries):")
print(pd.DataFrame(X_test.toarray(), columns=vectorizer.get_feature_names_out()).head())

print("\nTesting Labels (first 5 entries):")
print(y_test[:5])

# Convert the training and testing data to DataFrames for saving
X_train_df = pd.DataFrame(X_train.toarray(), columns=vectorizer.get_feature_names_out())
X_test_df = pd.DataFrame(X_test.toarray(), columns=vectorizer.get_feature_names_out())

# Create DataFrames for the labels
y_train_df = pd.DataFrame(y_train, columns=['tag'])
y_test_df = pd.DataFrame(y_test, columns=['tag'])

# Save the training and testing datasets to CSV files
X_train_df.to_csv("training_data.csv", index=False)
y_train_df.to_csv("training_labels.csv", index=False)
X_test_df.to_csv("testing_data.csv", index=False)
y_test_df.to_csv("testing_labels.csv", index=False)

print("Training and testing datasets saved as CSV files.")

# ---- ADD WORD CLOUD GENERATION CODE HERE ----
# Combine all processed patterns into a single string
all_patterns = ' '.join(texts)

# Generate the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_patterns)

# Display the word cloud using matplotlib
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
