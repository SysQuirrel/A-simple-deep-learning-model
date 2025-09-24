#!/usr/bin/env python3
import os

import optuna
from optuna.trial import TrialState
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

DEVICE = torch.device("cpu")
BATCHSIZE = 128
CLASSES = 3
DIR = os.getcwd()
EPOCHS = 500
N_TRAIN_EXAMPLES = BATCHSIZE * 30
N_VALID_EXAMPLES = BATCHSIZE * 10


def define_model(trial):
    # We optimize the number of layers, hidden units and dropout ratio in each layer.
    n_layers = trial.suggest_int("n_layers", 1, 16)
    layers = []

    in_features = 20
    for i in range(n_layers):
        out_features = trial.suggest_int("n_units_l{}".format(i), 4, 128)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        p = trial.suggest_float("dropout_l{}".format(i), 0.2, 0.5)
        layers.append(nn.Dropout(p))

        in_features = out_features
    layers.append(nn.Linear(in_features, CLASSES))
    layers.append(nn.LogSoftmax(dim=1))

    return nn.Sequential(*layers)

def load_stress_data():
    data = pd.read_csv("StressLevelDataset.csv")
    
    # Separate features and target
    X = data.drop(["stress_level"], axis=1)
    y = data["stress_level"].values
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split data - IMPORTANT: Only use train for training, valid for validation
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Fit only on training
    X_valid_scaled = scaler.transform(X_valid)      # Transform validation
    
    # Convert to tensors
    X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_valid = torch.tensor(X_valid_scaled, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_valid = torch.tensor(y_valid, dtype=torch.long)
    
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    valid_dataset = torch.utils.data.TensorDataset(X_valid, y_valid)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCHSIZE, shuffle=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=BATCHSIZE, shuffle=False
    )
    
    return train_loader, valid_loader

def objective(trial):
    # Generate the model.
    model = define_model(trial).to(DEVICE)

    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # Get the dataset.
    train_loader, valid_loader = load_stress_data()

    criterion = nn.CrossEntropyLoss()

    # Training of the model
    for epoch in range(EPOCHS):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        # Validation of the model
        model.eval()
        correct = 0
        total = 0
        val_loss = 0
        
        with torch.no_grad():
            for data, target in valid_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)

        accuracy = correct / total
        avg_val_loss = val_loss / len(valid_loader)

    
        trial.report(accuracy, epoch)
        
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50, timeout=1800)  # 30 minutes max

    print("Study statistics:")
    print("  Number of finished trials:", len(study.trials))
    
    print("Best trial:")
    trial = study.best_trial
    print("  Validation Accuracy:", trial.value)
    print("  Best Parameters:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")