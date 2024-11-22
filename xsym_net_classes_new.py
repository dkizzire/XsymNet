import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, feature_dim, num_heads):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(feature_dim, 4 * feature_dim),
            nn.ReLU(),
            nn.Linear(4 * feature_dim, feature_dim)
        )
        self.layernorm1 = nn.LayerNorm(feature_dim)
        self.layernorm2 = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # Attention part
        x_in = x.permute(0, 2, 1)  # Change shape for multihead attention
        attn_output, _ = self.attention(x_in, x_in, x_in)
        x = x + self.dropout(attn_output.permute(0, 2, 1))
        x = self.layernorm1(x)

        # Feed-forward part
        x_ff = self.feed_forward(x)
        x = x + self.dropout(x_ff)
        x = self.layernorm2(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.transformer_block = TransformerBlock(out_channels, num_heads=4)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.transformer_block(out)  # Transformer block after convolutions
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = F.relu(out)
        return out

class NN_bragg(nn.Module):
    def __init__(self, num_classes):
        super(NN_bragg, self).__init__()
        
        # Initial convolution layer for main input
        self.initial_conv_main = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )

        # Residual blocks for the main signal
        self.main_layer1 = ResidualBlock(64, 64)
        self.main_layer2 = ResidualBlock(64, 128, stride=2, downsample=nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm1d(128)
        ))
        self.main_layer3 = ResidualBlock(128, 256, stride=2, downsample=nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm1d(256)
        ))
        self.main_layer4 = ResidualBlock(256, 512, stride=2, downsample=nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm1d(512)
        ))

        # Initial convolution layer for peak positions
        self.initial_conv_peak = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )

        # Residual blocks for the peak positions
        self.peak_layer1 = ResidualBlock(64, 64)
        self.peak_layer2 = ResidualBlock(64, 128, stride=2, downsample=nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm1d(128)
        ))
        self.peak_layer3 = ResidualBlock(128, 256, stride=2, downsample=nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm1d(256)
        ))
        self.peak_layer4 = ResidualBlock(256, 512, stride=2, downsample=nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm1d(512)
        ))

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x, peak_positions):
        # Process main input
        x = self.initial_conv_main(x)
        x = self.main_layer1(x)
        x = self.main_layer2(x)
        x = self.main_layer3(x)
        x = self.main_layer4(x)

        # Process peak positions separately
        peak_x = self.initial_conv_peak(peak_positions)
        peak_x = self.peak_layer1(peak_x)
        peak_x = self.peak_layer2(peak_x)
        peak_x = self.peak_layer3(peak_x)
        peak_x = self.peak_layer4(peak_x)

        # Ensure dimensions match for attention mechanism
        if x.size(2) != peak_x.size(2):
            min_len = min(x.size(2), peak_x.size(2))
            x = x[:, :, :min_len]
            peak_x = peak_x[:, :, :min_len]

        # Apply attention using processed peak_positions to influence the main features
        x = x.permute(0, 2, 1)  # Change shape for attention
        peak_x = peak_x.permute(0, 2, 1)
        attention = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
        x, _ = attention(x, peak_x, peak_x)  # Using processed peak_x for key and value

        # Global pooling and final layers
        x = self.global_avg_pool(x.permute(0, 2, 1))  # Revert shape for pooling
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

# Setup training
model = NN_bragg(num_classes=10)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4) #L2 regularization with weight_decay

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, verbose=True)

# Example training loop
num_epochs = 50
train_loader = ...  # Define your DataLoader for training
val_loader = ...  # Define your DataLoader for validation

for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        x, peak_positions, y = batch
        optimizer.zero_grad()
        output = model(x, peak_positions)
        loss = F.cross_entropy(output, y)
        loss.backward()
        optimizer.step()

    # Validation step
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            x, peak_positions, y = batch
            output = model(x, peak_positions)
            val_loss += F.cross_entropy(output, y).item()

    val_loss /= len(val_loader)
    scheduler.step(val_loss)
    print(f'Epoch {epoch}, Validation Loss: {val_loss}')
