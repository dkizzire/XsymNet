import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

class simulated_data(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
# Define a class for a 1D Convolutional Neural Network
class BasicConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, dtype=torch.float, bias=False, **kwargs)
        self.bn = nn.BatchNorm1d(out_channels, dtype=torch.float)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(self.bn(x))
        return x
    
#^ ------------------------------------------------------------------------------------------------------
# * CNN_bragg - first good model 
#^ ------------------------------------------------------------------------------------------------------    


class CNN_bragg(nn.Module):
    def __init__(
            self,
            num_classes, 
            ):
        super().__init__()
        conv_block = BasicConv1d
        self.num_unique_classes = num_classes

        self.conv1 = conv_block(1, 256, kernel_size=31, stride=15, padding=15)

        self.maxpool_1 = nn.MaxPool1d(kernel_size=3, stride=2)

        self.conv2 = conv_block(256, 512, kernel_size=31, stride=15, padding=15)

        self.maxpool_2 = nn.MaxPool1d(kernel_size=3, stride=2)



        self.fc1 = nn.LazyLinear(1000)

        self.fc2 = nn.Linear(in_features=1000, out_features=num_classes)

    # define the forward method that applies the convolutional and fully connected layers to the input data
    def forward(self, x):
        x = x.unsqueeze(1)

        x = self.conv1(x)

        x = self.maxpool_1(x)

        x = self.conv2(x)

        x = self.maxpool_2(x)
    

        x = F.dropout1d(x, p=0.2)

        # Fully connected Linear
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))

        x = self.fc2(x)

        return x
    

#^ ------------------------------------------------------------------------------------------------------
# * CNN_simple - uses small kernals, used to compare to CNN_bragg 
#^ ------------------------------------------------------------------------------------------------------    

class CNN_simple(nn.Module):
    def __init__(
            self,
            num_classes, 
            ):
        super().__init__()
        conv_block = BasicConv1d
        self.num_unique_classes = num_classes

        self.conv1 = conv_block(1, 256, kernel_size=5, stride=1, padding=2)

        self.maxpool_1 = nn.MaxPool1d(kernel_size=3, stride=2)

        self.conv2 = conv_block(256, 512, kernel_size=3, stride=1, padding=1)

        self.maxpool_2 = nn.MaxPool1d(kernel_size=3, stride=2)



        self.fc1 = nn.LazyLinear(2000)
        self.fc2 = nn.LazyLinear(1000)
        self.fc3 = nn.LazyLinear(num_classes)

    # define the forward method that applies the convolutional and fully connected layers to the input data
    def forward(self, x):
        x = x.unsqueeze(1)
        
        x = self.conv1(x)

        x = self.maxpool_1(x)

        x = self.conv2(x)

        x = self.maxpool_2(x)

        x = F.dropout1d(x, p=0.2)

        # Fully connected Linear
        x = torch.flatten(x, 1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
    
#^ ------------------------------------------------------------------------------------------------------
# * AutoEncoder - does not work well for this application
#^ ------------------------------------------------------------------------------------------------------    

class XRD_Autoencoder(nn.Module):
    def __init__(self, input_size, encoded_dim):
        super(XRD_Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),  # More filters for peak detection
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 508, encoded_dim)  # Bottleneck layer
        # self.fc1 = nn.Linear(128 * ((input_size // 8)), encoded_dim)  # Bottleneck layer

        # Decoder
        self.fc2 = nn.Linear(encoded_dim, 128 * 508)
        # self.fc2 = nn.Linear(encoded_dim, 128 * (input_size // 8))
        self.unflatten = nn.Unflatten(1, (128, 508))
        # self.unflatten = nn.Unflatten(1, (128, input_size // 8))
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 1, kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.Sigmoid()  # Normalized output
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.encoder(x)
        # print(f'size of x after self.encoder = {x.size()}')
        x = self.flatten(x)
        # print(f'size of x after flatten = {x.size()}')
        encoded = self.fc1(x)  # Encoded feature vector
        encoded = encoded.squeeze(1)
        x = self.fc2(encoded)
        x = self.unflatten(x)
        x = x.squeeze(1)
        decoded = self.decoder(x)  # Reconstructed XRD pattern
        decoded = decoded.squeeze(1)
        return encoded, decoded

#^ ------------------------------------------------------------------------------------------------------
# * Simple Classifiear for Autoencoder setup
#^ ------------------------------------------------------------------------------------------------------    

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

#^ ------------------------------------------------------------------------------------------------------
# * Global Self Attention
#^ ------------------------------------------------------------------------------------------------------    

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.scale = torch.sqrt(torch.tensor(embed_dim, dtype=torch.float32))

    def forward(self, x):
        # Transpose to (batch, seq_len, embed_dim) if necessary
        x = x.permute(0, 2, 1)  
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attention_scores = torch.softmax(torch.bmm(Q, K.transpose(1, 2)) / self.scale, dim=-1)
        x = torch.bmm(attention_scores, V)
        
        # Restore shape to (batch, channels, seq_len)
        return x.permute(0, 2, 1)

class CNN_attention(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_unique_classes = num_classes

        self.conv1 = nn.Conv1d(1, 256, kernel_size=31, stride=15, padding=15)
        self.bn_1 = nn.BatchNorm1d(256, dtype=torch.float)
        self.maxpool_1 = nn.MaxPool1d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv1d(256, 512, kernel_size=31, stride=15, padding=15)
        self.bn_2 = nn.BatchNorm1d(512, dtype=torch.float)
        self.maxpool_2 = nn.MaxPool1d(kernel_size=3, stride=2)

        self.attention = SelfAttention(embed_dim=512)

        self.fc1 = nn.LazyLinear(1000)
        self.fc2 = nn.Linear(in_features=1000, out_features=num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.bn_1(x)
        x = self.maxpool_1(x)

        x = self.conv2(x)
        x = self.bn_2(x)
        x = self.maxpool_2(x)

        x = self.attention(x)  # Apply self-attention

        x = F.dropout1d(x, p=0.2)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


#^ ------------------------------------------------------------------------------------------------------
# * Local Self Attention
#^ ------------------------------------------------------------------------------------------------------    

class LocalSelfAttention(nn.Module):
    """
    Efficient Self-Attention for Long 1D Sequences (XRD)
    - Uses local attention with a sliding window (instead of full attention).
    """
    def __init__(self, embed_dim, window_size=30):
        super().__init__()
        self.window_size = window_size  # Local attention over a fixed window
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.scale = torch.sqrt(torch.tensor(embed_dim, dtype=torch.float32))

    def forward(self, x):
        # Reshape to (batch, seq_len, channels) for attention
        x = x.permute(0, 2, 1)  
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        batch, seq_len, dim = Q.shape
        attn_out = torch.zeros_like(Q)  # Placeholder for output

        # Compute attention within sliding windows
        for i in range(0, seq_len, self.window_size):
            start = max(0, i - self.window_size // 2)
            end = min(seq_len, i + self.window_size // 2)
            Q_win, K_win, V_win = Q[:, start:end], K[:, start:end], V[:, start:end]
            
            # Scaled Dot-Product Attention
            scores = torch.softmax(torch.bmm(Q_win, K_win.transpose(1, 2)) / self.scale, dim=-1)
            attn_out[:, start:end] = torch.bmm(scores, V_win)

        # Restore to (batch, channels, seq_len)
        return attn_out.permute(0, 2, 1)

class CNNWithAttention(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv1d(1, 256, kernel_size=31, stride=15, padding=15, bias=False)
        self.bn_1 = nn.BatchNorm1d(256, dtype=torch.float)
        self.maxpool_1 = nn.MaxPool1d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv1d(256, 512, kernel_size=31, stride=15, padding=15, bias=False)
        self.bn_2 = nn.BatchNorm1d(512, dtype=torch.float)
        self.maxpool_2 = nn.MaxPool1d(kernel_size=3, stride=2)

        # Attention applied after convolutional feature extraction
        self.attention = LocalSelfAttention(embed_dim=512, window_size=50)

        self.fc1 = nn.LazyLinear(1000)
        self.fc2 = nn.Linear(in_features=1000, out_features=num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension

        x = self.conv1(x)
        x = self.bn_1(x)
        x = self.maxpool_1(x)

        x = self.conv2(x)
        x = self.bn_2(x)
        x = self.maxpool_2(x)

        # Apply Local Self-Attention
        attn_x = self.attention(x)

        # Fusion: Add residual connection between CNN and Attention features
        x = x + attn_x  

        x = F.dropout1d(x, p=0.2)

        x = torch.flatten(x, 1)  # Flatten for classification
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

#^ ------------------------------------------------------------------------------------------------------
# * Mulit Head Self Attention
#^ ------------------------------------------------------------------------------------------------------   

class MultiheadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)  # Output projection

        self.scale = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

    def forward(self, x):
        batch_size, channels, seq_len = x.shape
        x = x.permute(0, 2, 1)  # Convert to (batch, seq_len, channels)

        # Linear projections
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch, heads, seq_len, head_dim)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        attn_output = self.output_proj(attn_output)

        return attn_output.permute(0, 2, 1)  # Convert back to (batch, channels, seq_len)


class CNN_MultiHeadAttention(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv1d(1, 256, kernel_size=31, stride=15, padding=15)
        self.maxpool_1 = nn.MaxPool1d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv1d(256, 512, kernel_size=31, stride=15, padding=15)
        self.maxpool_2 = nn.MaxPool1d(kernel_size=3, stride=2)

        # Attention applied after convolutional feature extraction
        self.attention = MultiheadSelfAttention(embed_dim=512, num_heads=2)

        self.fc1 = nn.LazyLinear(1000)
        self.fc2 = nn.Linear(in_features=1000, out_features=num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension

        x = self.conv1(x)
        x = self.maxpool_1(x)

        x = self.conv2(x)
        x = self.maxpool_2(x)

        # Apply Local Self-Attention
        attn_x = self.attention(x)

        # Fusion: Add residual connection between CNN and Attention features
        x = x + attn_x  

        x = F.dropout1d(x, p=0.3)

        x = torch.flatten(x, 1)  # Flatten for classification
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x