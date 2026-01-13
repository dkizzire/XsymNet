import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from matplotlib import rcParams
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
    

#^ ------------------------------------------------------------------------------------------------------
# * XsymNet with Multi-Head Attention
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

    def forward(self, x):
        batch_size, channels, seq_len = x.shape
        x = x.permute(0, 2, 1)  # (batch, seq_len, channels)

        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)

        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        attn_output = self.output_proj(attn_output)

        return attn_output.permute(0, 2, 1)


class CNN_MultiHeadAttention(nn.Module):
    def __init__(self, num_classes, channel_1, channel_2, kernel_1, kernel_2, stride_1, stride_2, num_heads, linear_1):
        super().__init__()

        # --- Make ALL input variables instance attributes ---
        self.num_classes = num_classes
        self.channel_1   = channel_1
        self.channel_2   = channel_2
        self.kernel_1    = kernel_1
        self.kernel_2    = kernel_2
        self.stride_1    = stride_1
        self.stride_2    = stride_2
        self.num_heads   = num_heads
        self.linear_1    = linear_1

        self.conv1 = nn.Conv1d(1, channel_1, kernel_size=kernel_1, stride=stride_1, padding=kernel_1 // 2, bias=False)
        self.bn1   = nn.BatchNorm1d(channel_1)
        self.maxpool1 = nn.MaxPool1d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv1d(channel_1, channel_2, kernel_size=kernel_2, stride=stride_2, padding=kernel_2 // 2, bias=False)
        self.bn2   = nn.BatchNorm1d(channel_2)
        self.maxpool2 = nn.MaxPool1d(kernel_size=3, stride=2)

        self.attention = MultiheadSelfAttention(embed_dim=channel_2, num_heads=num_heads)

        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.LazyLinear(linear_1)
        self.fc2 = nn.Linear(linear_1, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.maxpool2(x)

        attn_x = self.attention(x)
        x = x + attn_x

        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


#^ ------------------------------------------------------------------------------------------------------
# * XsymNet with Multi-Head Attention  :: Single Convolutional Layer
#^ ------------------------------------------------------------------------------------------------------   


class CNN_MultiHeadAttention_1Conv(nn.Module):
    def __init__(self, num_classes, channel_2, kernel_2, stride_2, num_heads, linear_1):
        super().__init__()
        self.num_classes = num_classes

        self.conv2 = nn.Conv1d(1, channel_2, kernel_size=kernel_2, stride=stride_2, padding=kernel_2 // 2, bias=False)
        self.bn2   = nn.BatchNorm1d(channel_2)
        self.maxpool2 = nn.MaxPool1d(kernel_size=3, stride=2)

        self.attention = MultiheadSelfAttention(embed_dim=channel_2, num_heads=num_heads)

        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.LazyLinear(linear_1)
        self.fc2 = nn.Linear(linear_1, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.maxpool2(x)

        attn_x = self.attention(x)
        x = x + attn_x

        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


#^ ------------------------------------------------------------------------------------------------------
# * Class Activation Map for specified target class
#^ ------------------------------------------------------------------------------------------------------   

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients: torch.Tensor | None = None
        self.activations: torch.Tensor | None = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)  # Use full backward hook

    def generate_cam(self, input_tensor, target_class: int):
        self.model.eval()
        output = self.model(input_tensor)
        self.model.zero_grad()
        class_score = output[:, target_class]
        class_score.backward()

        # --- Safety checks ---
        assert self.activations is not None, "Activations are None. Did the forward hook run?"
        assert self.gradients is not None, "Gradients are None. Did the backward hook run?"

        print(f"Activations Shape: {self.activations.shape}")  # Debugging
        print(f"Gradients Shape: {self.gradients.shape}")  # Debugging
        
        pooled_gradients = torch.mean(self.gradients, dim=[2])  # Global average pooling
        print(f"Pooled Gradients Shape: {pooled_gradients.shape}")  # Debugging
        
        cam = torch.zeros_like(self.activations[0, 0])
        for i in range(self.activations.shape[1]):
            cam += pooled_gradients[0, i] * self.activations[0, i]
        
        cam = F.relu(cam)  # Apply ReLU
        cam = cam - cam.min()
        cam = cam / cam.max()
        
        return cam.cpu().numpy()
    
    def generate_average_cam(self, input_batch, target_class):
        """
        Generate average CAM over a batch of samples from the same class.

        Args:
            input_batch (Tensor): Tensor of shape (N, seq_len), all from the same class.
            target_class (int): The target class index for CAM computation.

        Returns:
            avg_cam (np.ndarray): Averaged CAM over all input samples.
        """
        self.model.eval()
        cams = []

        for i in range(input_batch.shape[0]):
            input_tensor = input_batch[i:i+1]  # Keep batch dim
            output = self.model(input_tensor)
            self.model.zero_grad()

            class_score = output[:, target_class]
            class_score.backward(retain_graph=True)
            
            # --- Safety checks ---
            assert self.activations is not None, "Activations are None. Did the forward hook run?"
            assert self.gradients is not None, "Gradients are None. Did the backward hook run?"

            pooled_gradients = torch.mean(self.gradients, dim=[2])
            cam = torch.zeros_like(self.activations[0, 0])
            for j in range(self.activations.shape[1]):
                cam += pooled_gradients[0, j] * self.activations[0, j]

            cam = F.relu(cam)
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)
            cams.append(cam.cpu().numpy())

        avg_cam = np.mean(cams, axis=0)
        return avg_cam


#^ ------------------------------------------------------------------------------------------------------
# * Averaged Class Activation Map for specified target class
#^ ------------------------------------------------------------------------------------------------------   

def plot_avg_cam_with_expt(xrd_expt, avg_cam, x_axis, x_units='2θ (degrees)', save_name="avg_cam_plot.png", legend_label= '', title=None):
    
    avg_cam = avg_cam.squeeze()
    xrd_expt = xrd_expt.squeeze()
   
    rcParams.update({
        "text.usetex": False,  # Use True for LaTeX rendering if desired
        "font.family": "serif",
        "font.size": 16,
        "axes.labelsize": 16,
        "axes.titlesize": 14,
        "legend.fontsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "xtick.major.size": 10,
        "ytick.major.size": 10,
        "figure.dpi": 600,
        "savefig.bbox": "tight"
    })

    # Resize CAM if necessary
    if avg_cam.shape[0] != xrd_expt.shape[0]:
        print(f"Resizing CAM from {avg_cam.shape} to {xrd_expt.shape}")
        
        avg_cam = zoom(avg_cam, zoom=xrd_expt.shape[0] / avg_cam.shape[0])
        avg_cam = np.asarray(avg_cam)

    # Normalize CAM
    avg_cam = (avg_cam - np.min(avg_cam)) / (np.max(avg_cam) + 1e-8)

    # x-axis
    if len(x_axis) == 1:
        x_axis = np.arange(len(xrd_expt))

    fig, ax = plt.subplots(figsize=(12, 6), dpi=600)
    
    # Plot CAM as background
    ax.imshow(avg_cam[np.newaxis, :], extent=(x_axis[0], x_axis[-1], np.min(xrd_expt), np.max(xrd_expt)),
              cmap='jet', aspect='auto', alpha=0.8, origin='lower', vmin=avg_cam.min(), vmax=avg_cam.max())

    # Plot experimental XRD as black line
    ax.plot(x_axis, xrd_expt, color='black', linewidth=1.5, label=legend_label)

    if title:
        ax.set_title(title, fontsize=14)
    ax.set_xlabel(f'{x_units}', fontsize=14)
    ax.set_ylabel("Intensity A.U.", fontsize=14)
    ax.legend()
    plt.savefig(save_name, dpi=600, bbox_inches='tight')
    plt.show()

