import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import xsymnet_classes as xsym

# Load your real XRD dataset here
num_samples, num_features = 1000, 2000  # Adjust for real data
X_data = np.random.rand(num_samples, num_features)
y_labels = np.random.randint(0, 2, size=(num_samples,))  # Binary example

# Log transform to handle large intensity values
X_data = np.log1p(X_data)

# Min-Max Normalization
scaler = MinMaxScaler(feature_range=(0, 1))
X_data = scaler.fit_transform(X_data)

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X_data, y_labels, test_size=0.2, random_state=42)

# Convert to PyTorch tensors & reshape for Conv1D
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

# Create DataLoaders
batch_size = 32
train_loader = DataLoader(TensorDataset(X_train_tensor), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_tensor), batch_size=batch_size, shuffle=False)

# Initialize Autoencoder
encoded_dim = 100
model = xsym.XRD_Autoencoder(input_size=num_features, encoded_dim=encoded_dim)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Loss & optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch in train_loader:
        inputs = batch[0].to(device)
        optimizer.zero_grad()
        encoded, decoded = model(inputs)
        loss = criterion(decoded, inputs)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss / len(train_loader):.4f}")

print("Training complete.")

# Extract encoded features
with torch.no_grad():
    model.eval()
    X_train_encoded = torch.cat([model(batch[0].to(device))[0] for batch in train_loader]).cpu().numpy()
    X_val_encoded = torch.cat([model(batch[0].to(device))[0] for batch in val_loader]).cpu().numpy()

print("Encoded feature shape:", X_train_encoded.shape)

# Convert features to tensors for ML training
X_train_encoded_tensor = torch.tensor(X_train_encoded, dtype=torch.float32)
X_val_encoded_tensor = torch.tensor(X_val_encoded, dtype=torch.float32)

# Create DataLoader for ML model
train_ml_loader = DataLoader(TensorDataset(X_train_encoded_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
val_ml_loader = DataLoader(TensorDataset(X_val_encoded_tensor, y_val_tensor), batch_size=batch_size, shuffle=False)

# Simple Classifier on Encoded Features
class XRD_Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(XRD_Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),  # Increase hidden size for more complex patterns
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)  # Output layer now has `num_classes`
        )
    
    def forward(self, x):
        return self.fc(x)
# Train Classifier
classifier = XRD_Classifier(encoded_dim, 2).to(device)
criterion_cls = nn.CrossEntropyLoss()
optimizer_cls = optim.Adam(classifier.parameters(), lr=0.001)

num_epochs_cls = 10
for epoch in range(num_epochs_cls):
    classifier.train()
    train_loss_cls = 0
    for batch in train_ml_loader:
        inputs, labels = batch[0].to(device), batch[1].to(device)
        optimizer_cls.zero_grad()
        outputs = classifier(inputs)
        loss = criterion_cls(outputs, labels)
        loss.backward()
        optimizer_cls.step()
        train_loss_cls += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs_cls}], Classifier Loss: {train_loss_cls / len(train_ml_loader):.4f}")

print("Classifier training complete.")
