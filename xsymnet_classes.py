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
        self.num_classes = num_classes

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
    
    # def generate_average_cam(self, input_batch, target_class):
    #     """
    #     Generate average CAM over a batch of samples from the same class.

    #     Args:
    #         input_batch (Tensor): Tensor of shape (N, seq_len), all from the same class.
    #         target_class (int): The target class index for CAM computation.

    #     Returns:
    #         avg_cam (np.ndarray): Averaged CAM over all input samples.
    #     """
    #     self.model.eval()
    #     cams = []

    #     for i in range(input_batch.shape[0]):
    #         input_tensor = input_batch[i:i+1]  # Keep batch dim
    #         output = self.model(input_tensor)
    #         self.model.zero_grad()

    #         class_score = output[:, target_class]
    #         class_score.backward(retain_graph=True)

    #         pooled_gradients = torch.mean(self.gradients, dim=[2])
    #         cam = torch.zeros_like(self.activations[0, 0])
    #         for j in range(self.activations.shape[1]):
    #             cam += pooled_gradients[0, j] * self.activations[0, j]

    #         cam = F.relu(cam)
    #         cam = cam - cam.min()
    #         cam = cam / (cam.max() + 1e-8)
    #         cams.append(cam.cpu().numpy())

    #     avg_cam = np.mean(cams, axis=0)
    #     return avg_cam

    # def generate_average_cam(self, input_batch, target_class):
    #     """
    #     Generate average CAM over a batch of samples from the same class.
    #     """
    #     self.model.eval()
    #     cams = []

    #     for i in range(input_batch.shape[0]):
    #         input_tensor = input_batch[i:i+1]  # shape: (1, seq_len)
    #         self.model.zero_grad()
    #         output = self.model(input_tensor)

    #         # Get scalar class score by summing
    #         class_score = output[:, target_class].sum()
    #         class_score.backward(retain_graph=True)

    #         pooled_gradients = torch.mean(self.gradients, dim=[2])
    #         cam = torch.zeros_like(self.activations[0, 0])
    #         for j in range(self.activations.shape[1]):
    #             cam += pooled_gradients[0, j] * self.activations[0, j]

    #         cam = F.relu(cam)
    #         cam = cam - cam.min()
    #         cam = cam / (cam.max() + 1e-8)
    #         cams.append(cam.cpu().numpy())

    #     avg_cam = np.mean(cams, axis=0)
    #     return avg_cam

    def generate_average_cam(self, input_batch, target_class):
        """
        Generate average CAM over a batch of samples from the same class,
        in one forward/backward pass (faster than looping).
        """
        self.model.eval()
        self.model.zero_grad()

        # Forward pass (entire batch at once)
        output = self.model(input_batch)  # shape: (batch_size, num_classes)

        # Get scalar class score (sum over batch to backpropagate through all samples)
        class_score = output[:, target_class].sum()
        class_score.backward(retain_graph=False)  # gradients for all batch items

        # pooled_gradients: average gradient for each channel across batch + sequence
        # shape: (num_channels,)
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2])  

        # Weight activations with pooled gradients
        # self.activations shape: (batch_size, num_channels, seq_len)
        weighted_activations = (pooled_gradients[None, :, None] * self.activations).sum(dim=1)

        # Apply ReLU, normalize each CAM
        cams = F.relu(weighted_activations)
        cams = cams - cams.min(dim=1, keepdim=True)[0]
        cams = cams / (cams.max(dim=1, keepdim=True)[0] + 1e-8)

        # Average CAMs across the batch
        avg_cam = cams.mean(dim=0).cpu().numpy()  # shape: (seq_len,)

        return avg_cam


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
# * Averaged Class Activation Map for specified target class
#^ ------------------------------------------------------------------------------------------------------   




def plot_avg_cam_with_expt(xrd_expt, avg_cam, x_axis=None, x_units='2θ (degrees)', save_name="avg_cam_plot.png", legend_label= '', title=None):
    from matplotlib import rcParams
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
        from scipy.ndimage import zoom
        avg_cam = zoom(avg_cam, zoom=xrd_expt.shape[0] / avg_cam.shape[0])

    # Normalize CAM
    avg_cam = (avg_cam - avg_cam.min()) / (avg_cam.max() + 1e-8)

    # x-axis
    if x_axis is None:
        x_axis = np.arange(len(xrd_expt))

    fig, ax = plt.subplots(figsize=(12, 6), dpi=600)
    
    # Plot CAM as background
    ax.imshow(avg_cam[np.newaxis, :], extent=[x_axis[0], x_axis[-1], xrd_expt.min(), xrd_expt.max()],
              cmap='jet', aspect='auto', alpha=0.8, origin='lower', vmin=avg_cam.min(), vmax=avg_cam.max())

    # Plot experimental XRD as white line
    ax.plot(x_axis, xrd_expt, color='black', linewidth=1.5, label=legend_label)

    if title:
        ax.set_title(title, fontsize=14)
    ax.set_xlabel(f'{x_units}', fontsize=14)
    ax.set_ylabel("Intensity A.U.", fontsize=14)
    ax.legend()
    plt.savefig(save_name, dpi=600, bbox_inches='tight')
    plt.show()

