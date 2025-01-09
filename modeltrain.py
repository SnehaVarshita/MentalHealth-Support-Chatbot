import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import ast
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Tuple
import joblib
import os
from nltk.corpus import wordnet
import nltk
import random
from imblearn.over_sampling import RandomOverSampler

# Ensure NLTK WordNet is downloaded
nltk.download('wordnet')

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    # Convert 'responses' from string to list
    data['responses'] = data['responses'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])
    
    # Drop rows with missing 'patterns' and create a copy
    data = data.dropna(subset=['patterns']).copy()
    
    # Ensure 'patterns' are strings
    data['patterns'] = data['patterns'].astype(str)
    
    return data

def consolidate_intents(data):
    # Consolidate 'fact-*' intents
    fact_intents = [intent for intent in data['tag'].unique() if intent.startswith('fact-')]
    data['tag'] = data['tag'].apply(lambda x: 'fact' if x in fact_intents else x)
    
    # Consolidate other similar intents
    learn_intents = ['learn-more', 'learn', 'information']
    data['tag'] = data['tag'].apply(lambda x: 'learn' if x in learn_intents else x)
    
    return data

def handle_rare_intents(data, threshold=3):
    intent_counts = data['tag'].value_counts()
    rare_intents = intent_counts[intent_counts < threshold].index.tolist()
    data['tag'] = data['tag'].apply(lambda x: 'other' if x in rare_intents else x)
    return data

def synonym_replacement(sentence, n=1):
    words = sentence.split()
    new_words = words.copy()
    random_word_list = list(set([word for word in words if wordnet.synsets(word)]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = wordnet.synsets(random_word)
        if synonyms:
            synonym = synonyms[0].lemmas()[0].name().replace('_', ' ')
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break
    return ' '.join(new_words)

def augment_data(data, threshold=5, n_aug=2):
    augmented_patterns = []
    augmented_tags = []
    for tag, group in data.groupby('tag'):
        if len(group) < threshold:
            for _ in range(n_aug):
                for pattern in group['patterns']:
                    augmented_pattern = synonym_replacement(pattern)
                    augmented_patterns.append(augmented_pattern)
                    augmented_tags.append(tag)
    augmented_data = pd.DataFrame({'patterns': augmented_patterns, 'tag': augmented_tags})
    return pd.concat([data, augmented_data], ignore_index=True)

class IntentDataset(Dataset):
    def __init__(self, embeddings: np.ndarray, labels: np.ndarray):
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self) -> int:
        return len(self.embeddings)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.embeddings[idx], self.labels[idx]

class IntentClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: List[int], num_classes: int, dropout_rate: float = 0.3):
        super(IntentClassifier, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_classes))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

def train_model(model: nn.Module, 
                train_loader: DataLoader, 
                val_loader: DataLoader, 
                criterion: nn.Module, 
                optimizer: optim.Optimizer, 
                num_epochs: int, 
                device: torch.device,
                early_stopping_patience: int = 5) -> Dict:
    model = model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= early_stopping_patience:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    
    return history

def plot_training_history(history: Dict):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    # Load and preprocess data
    processed_data_path = './pdata.csv'
    data = load_data(processed_data_path)
    data = preprocess_data(data)
    
    # Apply preprocessing steps
    data = consolidate_intents(data)
    data = handle_rare_intents(data, threshold=3)
    data = augment_data(data, threshold=5, n_aug=2)
    
    # Get BERT embeddings
    bert_model = SentenceTransformer('bert-base-nli-mean-tokens')
    embeddings = bert_model.encode(data['patterns'].tolist(), convert_to_tensor=False)
    
    # Encode labels
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(data['tag'])
    
    # Handle class imbalance with Random Over Sampling
    ros = RandomOverSampler(random_state=42)
    embeddings_resampled, labels_resampled = ros.fit_resample(embeddings, labels)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings_resampled, labels_resampled, 
        test_size=0.2, random_state=42
    )
    
    # Create datasets and dataloaders
    train_dataset = IntentDataset(X_train, y_train)
    test_dataset = IntentDataset(X_test, y_test)
    
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize model
    input_size = embeddings.shape[1]
    hidden_sizes = [512, 256, 128]
    num_classes = len(label_encoder.classes_)
    
    model = IntentClassifier(input_size, hidden_sizes, num_classes)
    
    # Training parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # Train model
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=20,
        device=device,
        early_stopping_patience=5
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate model
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(
        all_labels,
        all_preds,
        target_names=label_encoder.classes_,
        zero_division=0
    ))
    
    # Save models and encoders
    models_dir = './models/'
    os.makedirs(models_dir, exist_ok=True)
    
    # Save model weights and config
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_size': input_size,
            'hidden_sizes': hidden_sizes,
            'num_classes': num_classes
        }
    }, os.path.join(models_dir, 'deep_learning_model.pth'))

    # Save label encoder, BERT model, and trained model in model_training.py
    joblib.dump(label_encoder, os.path.join(models_dir, 'label_encoder.joblib')) 
    joblib.dump(bert_model, os.path.join(models_dir, 'bert_encoder.joblib'))
    print(f"Model and encoders saved successfully in '{models_dir}'")

if __name__ == "__main__":
    main()