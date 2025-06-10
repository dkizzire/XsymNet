import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection



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
        self.conv = nn.Conv1d(in_channels, out_channels, dtype=torch.float, **kwargs)
        # self.conv = nn.Conv1d(in_channels, out_channels, dtype=torch.float, bias=False, **kwargs)
        self.bn = nn.BatchNorm1d(out_channels, dtype=torch.float)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(self.bn(x))
        return x
    
#^ ------------------------------------------------------------------------------------------------------
# * CNN_bragg - first good model 
#^ ------------------------------------------------------------------------------------------------------    


# class CNN_bragg(nn.Module):
#     def __init__(
#             self,
#             num_classes, 
#             ):
#         super().__init__()
#         conv_block = BasicConv1d
#         self.num_unique_classes = num_classes

#         self.conv1 = conv_block(1, 256, kernel_size=31, stride=1, padding=15)

#         self.maxpool_1 = nn.MaxPool1d(kernel_size=3, stride=2)

#         self.conv2 = conv_block(256, 512, kernel_size=31, stride=1, padding=15)

#         self.maxpool_2 = nn.MaxPool1d(kernel_size=3, stride=2)

#         self.dropout = nn.Dropout(0.2)

#         self.fc2 = nn.LazyLinear(1000)

#         self.fc3 = nn.Linear(in_features=1000, out_features=num_classes)

#     # define the forward method that applies the convolutional and fully connected layers to the input data
#     def forward(self, x):
#         x = x.unsqueeze(1)

#         x = self.conv1(x)

#         x = self.maxpool_1(x)

#         x = self.conv2(x)

#         x = self.maxpool_2(x)
    

#         # x = F.dropout1d(x, p=0.2)

#         # Fully connected Linear
#         x = torch.flatten(x, 1)
#         x = self.dropout(x)  
#         x = F.relu(self.fc2(x))

#         x = self.fc3(x)

#         return x
    
class CNN_bragg(nn.Module):
    def __init__(
            self,
            num_classes, 
            ):
        super().__init__()
        self.num_unique_classes = num_classes

        self.conv1 = nn.Conv1d(1, 256, kernel_size=31, stride=2, padding=15, bias=False)
        self.bn1 = nn.BatchNorm1d(256, dtype=torch.float)
        self.maxpool_1 = nn.MaxPool1d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv1d(256, 512, kernel_size=31, stride=2, padding=15, bias=False)
        self.bn2 = nn.BatchNorm1d(512, dtype=torch.float)
        self.maxpool_2 = nn.MaxPool1d(kernel_size=3, stride=2)

        # Dropout 
        self.dropout = nn.Dropout(0.2)

        # Linear
        self.fc1 = nn.LazyLinear(1000)
        self.fc2 = nn.Linear(in_features=1000, out_features=num_classes)

    # define the forward method that applies the convolutional and fully connected layers to the input data
    def forward(self, x):
        x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool_1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.maxpool_2(x)
    
        x = torch.flatten(x, 1)
        x = self.dropout(x)  
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
        self.num_unique_classes = num_classes

        self.conv1 = nn.Conv1d(1, 256, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(256, dtype=torch.float)
        self.maxpool_1 = nn.MaxPool1d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv1d(256, 512, kernel_size=5, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(512, dtype=torch.float)
        self.maxpool_2 = nn.MaxPool1d(kernel_size=3, stride=2)

        # Dropout 
        self.dropout = nn.Dropout(0.2)

        # Linear
        self.fc1 = nn.LazyLinear(1000)
        self.fc2 = nn.Linear(in_features=1000, out_features=num_classes)

    # define the forward method that applies the convolutional and fully connected layers to the input data
    def forward(self, x):
        x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool_1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.maxpool_2(x)
    
        x = torch.flatten(x, 1)
        x = self.dropout(x) 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    


#^ ------------------------------------------------------------------------------------------------------
# * Global Self Attention
#^ ------------------------------------------------------------------------------------------------------    

class GlobalSelfAttention(nn.Module):
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

class CNN_GlobalSelfAttention(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_unique_classes = num_classes

        self.conv1 = nn.Conv1d(1, 256, kernel_size=31, stride=2, padding=15, bias=False)
        self.bn_1 = nn.BatchNorm1d(256, dtype=torch.float)
        self.maxpool_1 = nn.MaxPool1d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv1d(256, 512, kernel_size=31, stride=2, padding=15, bias=False)
        self.bn_2 = nn.BatchNorm1d(512, dtype=torch.float)
        self.maxpool_2 = nn.MaxPool1d(kernel_size=3, stride=2)

        self.attention = GlobalSelfAttention(embed_dim=512)
        # Dropout performed after flattening
        self.dropout = nn.Dropout(0.2)
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


        x = torch.flatten(x, 1)
        x = self.dropout(x)  # Correctly apply dropout here
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

class CNN_LocalAttention(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv1d(1, 256, kernel_size=31, stride=2, padding=15)
        # self.bn_1 = nn.BatchNorm1d(256, dtype=torch.float)
        self.maxpool_1 = nn.MaxPool1d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv1d(256, 512, kernel_size=31, stride=2, padding=15)
        # self.bn_2 = nn.BatchNorm1d(512, dtype=torch.float)
        self.maxpool_2 = nn.MaxPool1d(kernel_size=3, stride=2)

        # Attention applied after convolutional feature extraction
        self.attention = LocalSelfAttention(embed_dim=512, window_size=50)

        # Dropout performed after flattening
        self.dropout = nn.Dropout(0.2)

        self.fc1 = nn.LazyLinear(1000)
        self.fc2 = nn.Linear(in_features=1000, out_features=num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension

        x = self.conv1(x)
        # x = self.bn_1(x)
        x = F.relu(x)
        x = self.maxpool_1(x)

        x = self.conv2(x)
        # x = self.bn_2(x)
        x = F.relu(x)
        x = self.maxpool_2(x)

        # Apply Local Self-Attention
        attn_x = self.attention(x)

        # Fusion: Add residual connection between CNN and Attention features
        x = x + attn_x  

        x = torch.flatten(x, 1)  # Flatten for classification
        x = self.dropout(x)  # Correctly apply dropout here
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


#^ ------------------------------------------------------------------------------------------------------
# * Mulit Head Self Attention - with batchnormalization -- & attn dropout
#^ ------------------------------------------------------------------------------------------------------   
class MultiheadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, attn_dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key   = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self.scale = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        # Dropout for attention weights (improves regularization)
        # self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(self, x):
        batch_size, channels, seq_len = x.shape
        x = x.permute(0, 2, 1)  # (batch, seq_len, channels)

        # Linear projections and reshape for multi-head attention
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        # attn_weights = self.attn_dropout(attn_weights)  # dropout on attention weights

        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        attn_output = self.output_proj(attn_output)

        return attn_output.permute(0, 2, 1)  # Back to (batch, channels, seq_len)

class CNN_MultiHeadAttention(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        channel_1 = 8
        channel_2 = 16
        kernel_1 = 31
        kernel_2 = 31
        # First convolutional block
        self.conv1 = nn.Conv1d(1, channel_1, kernel_size=kernel_1, stride=2, padding=kernel_1 // 2, bias=False)
        self.bn1   = nn.BatchNorm1d(channel_1)
        self.maxpool1 = nn.MaxPool1d(kernel_size=3, stride=2)

        # Second convolutional block
        self.conv2 = nn.Conv1d(channel_1, channel_2, kernel_size=kernel_2, stride=2, padding=kernel_2 // 2, bias=False)
        self.bn2   = nn.BatchNorm1d(channel_2)
        self.maxpool2 = nn.MaxPool1d(kernel_size=3, stride=2)

        # Multi-head self-attention applied to conv features
        self.attention = MultiheadSelfAttention(embed_dim=channel_2, num_heads=4)

        # Dropout performed after flattening
        self.dropout = nn.Dropout(0.2)

        # Fully connected layers for classification
        self.fc1 = nn.LazyLinear(1000)
        self.fc2 = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension: (batch, 1, seq_len)

        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool1(x)

        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.maxpool2(x)

        # Apply multi-head attention
        attn_x = self.attention(x)

        # Fusion: Residual connection: fuse CNN and attention features
        x = x + attn_x  

        # Flatten the features for classification
        x = torch.flatten(x, 1)
        x = self.dropout(x)  # Correctly apply dropout here
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

'''
-- x init shape  [batch_size, features]
torch.Size([512, 6932]) 
unsqueeze -- [batch_size, channels, features]
torch.Size([512, 1, 6932])
self.conv1 -- k=31, s=2, p=15
torch.Size([512, 8, 3466])
self.maxpool1 -- k=3, s=2
torch.Size([512, 8, 1732])

self.conv2 -- k=31, s=2, p=15
torch.Size([512, 16, 866])
self.maxpool2 -- k=3, s=2
torch.Size([512, 16, 432])

attn_x -- torch.Size([512, 16, 432])

x = x + attn_x
torch.Size([512, 16, 432])

flatten
torch.Size([512, 6912])
dropout
torch.Size([512, 6912])
self.fc1 -- Linear(in_features=6912, out_features=1000, bias=True)
torch.Size([512, 1000])
self.fc2 -- Linear(in_features=1000, out_features=545, bias=True)
torch.Size([512, 545])
'''
#^ ------------------------------------------------------------------------------------------------------
# * Mulit Head Self Attention - with hyperparameter tuning
#^ ------------------------------------------------------------------------------------------------------   

class MultiheadSelfAttention_hyp(nn.Module):
    def __init__(self, embed_dim, num_heads=4, attn_dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key   = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self.scale = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        # Dropout for attention weights (improves regularization)
        self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(self, x):
        batch_size, channels, seq_len = x.shape
        x = x.permute(0, 2, 1)  # (batch, seq_len, channels)

        # Linear projections and reshape for multi-head attention
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)  # dropout on attention weights

        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        attn_output = self.output_proj(attn_output)

        return attn_output.permute(0, 2, 1)  # Back to (batch, channels, seq_len)

class CNN_MultiHeadAttention_hyp(nn.Module):
    def __init__(self, num_classes, kernel1, stride1, kernel2, stride2):
        super().__init__()
        self.num_classes = num_classes

        # First convolutional block
        self.conv1 = nn.Conv1d(1, 256, kernel_size=kernel1, stride=stride1, padding=kernel1 // 2)
        self.bn1   = nn.BatchNorm1d(256)
        self.maxpool1 = nn.MaxPool1d(kernel_size=3, stride=2)

        # Second convolutional block
        self.conv2 = nn.Conv1d(256, 512, kernel_size=kernel2, stride=stride2, padding=kernel2 // 2)
        self.bn2   = nn.BatchNorm1d(512)
        self.maxpool2 = nn.MaxPool1d(kernel_size=3, stride=2)

        # Multi-head self-attention applied to conv features
        self.attention = MultiheadSelfAttention_hyp(embed_dim=512, num_heads=4)

        # Dropout defined as a module for use after flattening
        self.dropout = nn.Dropout(0.2)

        # Fully connected layers for classification
        self.fc1 = nn.LazyLinear(1000)
        self.fc2 = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension: (batch, 1, seq_len)

        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool1(x)

        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.maxpool2(x)

        # Apply multi-head attention
        attn_x = self.attention(x)

        # Residual connection: fuse CNN and attention features
        x = x + attn_x  

        # Flatten the features for classification
        x = torch.flatten(x, 1)
        x = self.dropout(x) 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#^ ------------------------------------------------------------------------------------------------------
# * AutoEncoder - does not work well for this application
#^ ------------------------------------------------------------------------------------------------------    

# class XRD_Autoencoder(nn.Module):
#     def __init__(self, input_size, encoded_dim):
#         super(XRD_Autoencoder, self).__init__()
#         # Encoder
#         self.encoder = nn.Sequential(
#             nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),  # More filters for peak detection
#             nn.ReLU(),
#             nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
#             nn.ReLU(),
#             nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
#             nn.ReLU()
#         )
#         self.flatten = nn.Flatten()
#         self.fc1 = nn.Linear(128 * 508, encoded_dim)  # Bottleneck layer
#         # self.fc1 = nn.Linear(128 * ((input_size // 8)), encoded_dim)  # Bottleneck layer

#         # Decoder
#         self.fc2 = nn.Linear(encoded_dim, 128 * 508)
#         # self.fc2 = nn.Linear(encoded_dim, 128 * (input_size // 8))
#         self.unflatten = nn.Unflatten(1, (128, 508))
#         # self.unflatten = nn.Unflatten(1, (128, input_size // 8))
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=0),
#             nn.ReLU(),
#             nn.ConvTranspose1d(32, 1, kernel_size=7, stride=2, padding=3, output_padding=1),
#             nn.Sigmoid()  # Normalized output
#         )

#     def forward(self, x):
#         x = x.unsqueeze(1)
#         x = self.encoder(x)
#         # print(f'size of x after self.encoder = {x.size()}')
#         x = self.flatten(x)
#         # print(f'size of x after flatten = {x.size()}')
#         encoded = self.fc1(x)  # Encoded feature vector
#         encoded = encoded.squeeze(1)
#         x = self.fc2(encoded)
#         x = self.unflatten(x)
#         x = x.squeeze(1)
#         decoded = self.decoder(x)  # Reconstructed XRD pattern
#         decoded = decoded.squeeze(1)
#         return encoded, decoded

# #^ ------------------------------------------------------------------------------------------------------
# # * Simple Classifiear for Autoencoder setup
# #^ ------------------------------------------------------------------------------------------------------    

# # Simple Classifier on Encoded Features
# class XRD_Classifier(nn.Module):
#     def __init__(self, input_dim, num_classes):
#         super(XRD_Classifier, self).__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(input_dim, 128),  # Increase hidden size for more complex patterns
#             nn.ReLU(),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, num_classes)  # Output layer now has `num_classes`
#         )
    
#     def forward(self, x):
#         return self.fc(x)


#^ ------------------------------------------------------------------------------------------------------
# * Class Activation Map for specified target class
#^ ------------------------------------------------------------------------------------------------------   

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)  # Use full backward hook

    def generate_cam(self, input_tensor, target_class):
        self.model.eval()
        output = self.model(input_tensor)
        self.model.zero_grad()
        class_score = output[:, target_class]
        class_score.backward()

        print(f"Activations Shape: {self.activations.shape}")  # Debugging
        print(f"Gradients Shape: {self.gradients.shape}")  # Debugging
        
        pooled_gradients = torch.mean(self.gradients, dim=[2])  # Global average pooling
        print(f"Pooled Gradients Shape: {pooled_gradients.shape}")  # Debugging
        
        cam = torch.zeros_like(self.activations[0, 0])
        for i in range(self.activations.shape[1]):
            cam += pooled_gradients[0, i] * self.activations[0, i]
        
        cam = F.relu(cam)  # Apply ReLU
        
        # if cam.ndim < 2:  # If 1D, reshape it
        #     print(f"Warning: CAM shape {cam.shape}, reshaping to (1, -1)")
        #     cam = cam.view(1, -1)

        cam = cam - cam.min()
        cam = cam / cam.max()
        
        return cam.cpu().numpy()



def plot_cam(xrd_pattern, cam, x_axis=None, x_units='ToF (μs)', save_name="grad_cam_plot.png", title=None, overlay=True, dpi=600):

    # quick x_units for pasting:
    # 2θ (degrees), ToF (μs), d-spacing (Å)

    cam = cam.squeeze()
    xrd_pattern = xrd_pattern.squeeze()

    if cam.shape[0] != xrd_pattern.shape[0]:
        print(f"Resizing CAM from {cam.shape} to {xrd_pattern.shape}")
        from scipy.ndimage import zoom
        cam = zoom(cam, zoom=xrd_pattern.shape[0] / cam.shape[0])

    # Normalize CAM for consistent coloring
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    # Use custom x-axis
    if x_axis is None:
        x_axis = np.arange(len(xrd_pattern))

    fig, ax = plt.subplots(figsize=(12, 6), dpi=dpi)  # Set DPI and figsize

    if overlay:
        # Use LineCollection for smooth color transitions
        norm = mcolors.Normalize(vmin=0, vmax=1)
        cmap = plt.cm.jet

        # Create line segments
        points = np.array([x_axis, xrd_pattern]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=1)  # Set LINE WIDTH
        lc.set_array(cam)

        ax.add_collection(lc)
        ax.autoscale()

        # Add colorbar
        fig.colorbar(lc, ax=ax, label="Activation Intensity")

    else:
        ax.plot(x_axis, xrd_pattern, label="XRD Pattern", color="black")
        scatter = ax.scatter(x_axis, xrd_pattern, c=cam, cmap="jet", alpha=0.5)
        fig.colorbar(scatter, ax=ax, label="Activation Intensity")
    
    if title is not None:
        ax.set_title(title, fontsize=14) 
    
    ax.set_xlabel(f'{x_units}', fontsize=12)  
    ax.set_ylabel("Intensity A.U.", fontsize=12)
    ax.legend()
    
    plt.savefig(save_name, dpi=dpi, bbox_inches='tight')  # High-resolution save
    plt.show()


#^ ------------------------------------------------------------------------------------------------------
# * Mulit Head Self Attention - with batchnormalization -- & attn dropout
#^ ------------------------------------------------------------------------------------------------------   
class MultiheadSelfAttention_fft(nn.Module):
    def __init__(self, embed_dim, num_heads=4, attn_dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key   = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self.scale = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        # Dropout for attention weights (improves regularization)
        # self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(self, x):
        batch_size, channels, seq_len = x.shape
        x = x.permute(0, 2, 1)  # (batch, seq_len, channels)

        # Linear projections and reshape for multi-head attention
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        # attn_weights = self.attn_dropout(attn_weights)  # dropout on attention weights

        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        attn_output = self.output_proj(attn_output)

        return attn_output.permute(0, 2, 1)  # Back to (batch, channels, seq_len)

class CNN_MultiHeadAttention_fft(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        channel_1 = 8
        channel_2 = 16
        kernel_1 = 3
        kernel_2 = 3
        # First convolutional block
        self.conv1 = nn.Conv1d(1, channel_1, kernel_size=kernel_1, stride=2, padding=kernel_1 // 2, bias=False)
        self.bn1   = nn.BatchNorm1d(channel_1)
        self.maxpool1 = nn.MaxPool1d(kernel_size=3, stride=2)

        # Second convolutional block
        self.conv2 = nn.Conv1d(channel_1, channel_2, kernel_size=kernel_2, stride=2, padding=kernel_2 // 2, bias=False)
        self.bn2   = nn.BatchNorm1d(channel_2)
        self.maxpool2 = nn.MaxPool1d(kernel_size=3, stride=2)

        # Multi-head self-attention applied to conv features
        self.attention = MultiheadSelfAttention_fft(embed_dim=channel_2, num_heads=4)

        # Dropout performed after flattening
        self.dropout = nn.Dropout(0.2)

        # Fully connected layers for classification
        self.fc1 = nn.LazyLinear(1000)
        self.fc2 = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension: (batch, 1, seq_len)

        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool1(x)

        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.maxpool2(x)

        # Apply multi-head attention
        attn_x = self.attention(x)

        # Fusion: Residual connection: fuse CNN and attention features
        x = x + attn_x  

        # Flatten the features for classification
        x = torch.flatten(x, 1)
        x = self.dropout(x)  # Correctly apply dropout here
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    


