# chatbot_utils.py
# type: ignore
import joblib
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
import numpy as np
import random
from modeltrain import IntentClassifier 

# Load necessary files
label_encoder = joblib.load('./models/label_encoder.joblib')
bert_model = joblib.load('./models/bert_encoder.joblib')

# Load model configuration and weights
model_checkpoint = torch.load('./models/deep_learning_model.pth')
model = IntentClassifier(
    input_size=model_checkpoint['model_config']['input_size'],
    hidden_sizes=model_checkpoint['model_config']['hidden_sizes'],
    num_classes=model_checkpoint['model_config']['num_classes']
)
model.load_state_dict(model_checkpoint['model_state_dict'])
model.eval()

# Load responses from processed_data.csv
data = pd.read_csv('./pdata.csv')
responses_dict = data.groupby('tag')['responses'].apply(lambda x: x.iloc[0]).to_dict()

# Function to predict the class (intent) of user input
def pred_class(message, bert_model, model, label_encoder):
    embedding = bert_model.encode([message])
    tensor_embedding = torch.FloatTensor(embedding)
    with torch.no_grad():
        output = model(tensor_embedding)
        _, predicted = torch.max(output, 1)
    predicted_tag = label_encoder.inverse_transform(predicted.numpy())[0]
    return predicted_tag

# Function to get response based on predicted intent
def get_response(intent, responses_dict):
    responses = eval(responses_dict.get(intent, "[]"))
    return random.choice(responses) if responses else "I'm here to listen. Tell me more about how you're feeling."