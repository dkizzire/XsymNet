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
        self.scale = self.head_dim ** 0.5

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key   = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        # self.scale = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

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
    def __init__(self, num_classes, channel_1, channel_2, kernel_1, kernel_2, stride_1, stride_2, num_heads, linear_1, gn_groups_1=8, gn_groups_2=8):
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

        # ---- CNN trunk ----
        self.conv1 = nn.Conv1d(1, channel_1, kernel_size=kernel_1, stride=stride_1, padding=kernel_1 // 2, bias=False)
        # self.bn1   = nn.BatchNorm1d(channel_1)
        self.gn1 = nn.GroupNorm(num_groups=gn_groups_1, num_channels=channel_1)
        self.maxpool1 = nn.MaxPool1d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv1d(channel_1, channel_2, kernel_size=kernel_2, stride=stride_2, padding=kernel_2 // 2, bias=False)
        # self.bn2   = nn.BatchNorm1d(channel_2)
        self.gn2 = nn.GroupNorm(num_groups=gn_groups_2, num_channels=channel_2)
        self.maxpool2 = nn.MaxPool1d(kernel_size=3, stride=2)
        
        # ---- Multi-Head Attention Layer ----
        self.ln_attention = nn.LayerNorm(channel_2)
        self.attention = MultiheadSelfAttention(embed_dim=channel_2, num_heads=num_heads)

        # ---- Classifier ----
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.LazyLinear(linear_1)
        self.fc2 = nn.Linear(linear_1, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)

        # x = F.relu(self.bn1(self.conv1(x)))
        # x = F.leaky_relu(self.bn1(self.conv1(x)))
        # x = self.maxpool1(x)

        # x = F.relu(self.bn2(self.conv2(x)))
        # x = F.leaky_relu(self.bn2(self.conv2(x)))
        # x = self.maxpool2(x)
        
        # attn_x = self.attention(x)
        # x = x + attn_x

        # x = torch.flatten(x, 1)
        # x = self.dropout(x)
        # x = F.relu(self.fc1(x))
        
        # ---- CNN trunk ----
        x = self.conv1(x)
        x = self.gn1(x)
        x = F.relu(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.gn2(x)
        x = F.relu(x)
        x = self.maxpool2(x)

        # ---- Layer Norm for Attention Stability ----
        residual = x
        x_ln = x.permute(0, 2, 1)  # (batch, seq_len, channels)
        x_ln = self.ln_attention(x_ln)
        x_ln = x_ln.permute(0, 2, 1)  # (batch, channels, seq_len)

        # ---- Multi-Head Attention Layer ----
        attn_x = self.attention(x_ln)
        x = residual + attn_x  # Residual connection

        # ---- Classifier ----
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))

        return self.fc2(x)

#^ ------------------------------------------------------------------------------------------------------
# * XsymNet with Multi-Head Attention  :: Multi-Channel Input (3 channels)
#^ ------------------------------------------------------------------------------------------------------

class CNN_MHA_MultiChannelInput(nn.Module):
    """CNN with Multi-Head Attention for 3-channel input.

    Accepts pre-computed multi-channel input:
        Channel 1: Intensity (MinMax-scaled)
        Channel 2: Log-scaled intensity
        Channel 3: Curvature (second derivative)

    Input shape: (batch, 3, seq_len)
    """

    def __init__(self, num_classes, channel_1, channel_2, kernel_1, kernel_2,
                 stride_1, stride_2, num_heads, linear_1, in_channels=3):
        super().__init__()

        # Store hyperparameters as instance attributes
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.channel_1   = channel_1
        self.channel_2   = channel_2
        self.kernel_1    = kernel_1
        self.kernel_2    = kernel_2
        self.stride_1    = stride_1
        self.stride_2    = stride_2
        self.num_heads   = num_heads
        self.linear_1    = linear_1

        # First conv block - takes in_channels (default 3) instead of 1
        self.conv1 = nn.Conv1d(in_channels, channel_1, kernel_size=kernel_1,
                               stride=stride_1, padding=kernel_1 // 2, bias=False)
        self.bn1   = nn.BatchNorm1d(channel_1)
        self.maxpool1 = nn.MaxPool1d(kernel_size=3, stride=2)

        # Second conv block
        self.conv2 = nn.Conv1d(channel_1, channel_2, kernel_size=kernel_2,
                               stride=stride_2, padding=kernel_2 // 2, bias=False)
        self.bn2   = nn.BatchNorm1d(channel_2)
        self.maxpool2 = nn.MaxPool1d(kernel_size=3, stride=2)

        # Multi-head self-attention
        self.attention = MultiheadSelfAttention(embed_dim=channel_2, num_heads=num_heads)

        # Classification head
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.LazyLinear(linear_1)
        self.fc2 = nn.Linear(linear_1, num_classes)

    def forward(self, x):
        # Input: (batch, 3, seq_len) - NO unsqueeze needed!

        # First conv block
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool1(x)

        # Second conv block
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.maxpool2(x)

        # Attention with residual connection
        attn_x = self.attention(x)
        x = x + attn_x

        # Classification
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
        # cam = cam - cam.min()
        # cam = cam / cam.max()
        
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
        print(f"Generating CAM for batch of size {input_batch.shape[0]} targeting class {target_class}...")  # Debugging
        for i in range(input_batch.shape[0]):
            input_tensor = input_batch[i:i+1]  # Keep batch dim
            output = self.model(input_tensor)
            self.model.zero_grad()

            class_score = output[0, target_class]  # Must be scalar for backward() - maybe replace with output[i, target_class] if batch processing is needed
            class_score.backward()

            
            # --- Safety checks ---
            assert self.activations is not None, "Activations are None. Did the forward hook run?"
            assert self.gradients is not None, "Gradients are None. Did the backward hook run?"

            pooled_gradients = torch.mean(self.gradients, dim=[2])
            cam = torch.zeros_like(self.activations[0, 0])
            for j in range(self.activations.shape[1]):
                cam += pooled_gradients[0, j] * self.activations[0, j]

            cam = F.relu(cam)
            # cam = cam - cam.min()
            # cam = cam / (cam.max() + 1e-8)
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
        # "text.usetex": True,  # Use True for LaTeX rendering if desired
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
    # plt.show()
    plt.close()

def plot_avg_cam_with_expt_zoom_in(xrd_expt, avg_cam, x_axis, x_units='2θ (degrees)', save_name="avg_cam_plot.png", legend_label= '', title=None):
    
    avg_cam = avg_cam.squeeze()
    xrd_expt = xrd_expt.squeeze()
   
    rcParams.update({
        # "text.usetex": True,  # Use True for LaTeX rendering if desired
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

    # # Resize CAM if necessary
    # if avg_cam.shape[0] != xrd_expt.shape[0]:
    #     print(f"Resizing CAM from {avg_cam.shape} to {xrd_expt.shape}")
        
    #     avg_cam = zoom(avg_cam, zoom=xrd_expt.shape[0] / avg_cam.shape[0])
    #     avg_cam = np.asarray(avg_cam)

    # Resize Experiment if necessary
    if avg_cam.shape[0] != xrd_expt.shape[0]:
        print(f"Resizing Expt from {xrd_expt.shape} to {avg_cam.shape} ")
        print(f"Resizing Expt x_axis from {x_axis.shape} to {avg_cam.shape} ")
        
        xrd_expt = zoom(xrd_expt, zoom= avg_cam.shape[0] / xrd_expt.shape[0])
        xrd_expt = np.asarray(xrd_expt)
        x_axis = zoom(x_axis, zoom= avg_cam.shape[0] / x_axis.shape[0])
        x_axis = np.asarray(x_axis)


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
    # plt.show()
    plt.close()


    
#^ ------------------------------------------------------------------------------------------------------
# * Integrated Gradients
#^ ------------------------------------------------------------------------------------------------------

class IntegratedGradients:
    """
    Integrated Gradients (IG) attributions at full input resolution.

    Reference: Sundararajan et al., "Axiomatic Attribution for Deep Networks" (2017).

    Unlike GradCAM (which hooks a downsampled intermediate feature map), IG
    operates directly at the raw input level: attributions have the same length
    as the input diffraction pattern with no upsampling artifacts.  The
    attention layer's global gradient routing does not corrupt IG because
    gradients are accumulated at the input rather than at an intermediate
    feature map.

    Completeness axiom (built-in correctness guarantee):
        Σᵢ IG_i(x) ≈ F_c(x) − F_c(x′)
    Attributions sum to the model's prediction gap between the actual input
    and the baseline, so the budget is exactly preserved.

    Args:
        model (nn.Module): Trained XSymNet model (plain or DataParallel-wrapped).
    """

    def __init__(self, model):
        self.model = model

    def generate_ig(self, input_tensor, target_class, baseline=None, n_steps=50):
        """
        Compute Integrated Gradients for a single input sample.

        The integral ∫₀¹ (∂F_c / ∂x)|_{x′ + α(x−x′)} dα is approximated with a
        Riemann sum over n_steps evenly spaced α values in [0, 1].

        torch.autograd.grad is used instead of .backward() so that model
        parameter gradients are never touched — only path_inputs receives a
        gradient, leaving the model state clean across repeated calls.

        Args:
            input_tensor (Tensor): shape (1, seq_len). Input diffraction pattern.
            target_class (int): Class index for which attributions are computed.
            baseline (Tensor | None): shape (1, seq_len). The reference
                "uninformative" input. Defaults to all-zeros, which is physically
                natural for MinMax-scaled data (zero intensity at every bin).
                For z-score normalised data, all-zeros ≡ average intensity.
            n_steps (int): Riemann-sum approximation steps. 50 is practical for
                exploratory work; use 200-300 for final publication figures.

        Returns:
            ig (np.ndarray): shape (seq_len,). Signed attribution per TOF/2θ bin.
                Positive → supports the target-class prediction.
                Negative → opposes it (pushes the model toward other classes).
        """
        self.model.eval()

        if baseline is None:
            baseline = torch.zeros_like(input_tensor)

        # Straight-line interpolation path: baseline (α=0) → actual input (α=1)
        alphas     = torch.linspace(0.0, 1.0, n_steps, device=input_tensor.device)
        delta      = (input_tensor - baseline).detach()                      # (1, seq_len)
        path_inputs = baseline.detach() + alphas.view(-1, 1) * delta         # (n_steps, seq_len)
        path_inputs = path_inputs.requires_grad_(True)                       # leaf tensor

        outputs = self.model(path_inputs)                                    # (n_steps, num_classes)
        score   = outputs[:, target_class].sum()

        # Compute gradient only w.r.t. path_inputs; model parameter grads untouched
        grads     = torch.autograd.grad(score, path_inputs)[0]               # (n_steps, seq_len)
        avg_grads = grads.detach().mean(dim=0)                               # (seq_len,)
        ig        = delta.squeeze() * avg_grads                              # (seq_len,)
        return ig.cpu().numpy()

    def generate_average_ig(self, input_batch, target_class, baseline=None, n_steps=50):
        """
        Average IG attributions over a batch of samples from the same class.

        Analogous to GradCAM's generate_average_cam: averages attributions over
        N simulated patterns of the target subgroup to produce a class-level map.

        Computational note: this performs n_steps forward passes per sample.
        For large batches (> a few hundred), pass a subsampled slice of
        input_batch or reduce n_steps to manage runtime.

        Args:
            input_batch (Tensor): shape (N, seq_len). All from the same subgroup.
            target_class (int): Class index for attribution.
            baseline (Tensor | None): shape (1, seq_len). Defaults to all-zeros.
            n_steps (int): Integration steps per sample.

        Returns:
            avg_ig (np.ndarray): shape (seq_len,). Mean signed attribution per bin.
        """
        if baseline is None:
            baseline = torch.zeros(1, input_batch.shape[1], device=input_batch.device)

        igs = []
        print(f"Generating IG for {input_batch.shape[0]} samples → class {target_class}...")
        for i in range(input_batch.shape[0]):
            ig = self.generate_ig(
                input_batch[i:i+1], target_class,
                baseline=baseline, n_steps=n_steps
            )
            igs.append(ig)

        return np.mean(igs, axis=0)   # (seq_len,)


#^ ------------------------------------------------------------------------------------------------------
# * Attention Rollout
#^ ------------------------------------------------------------------------------------------------------

class AttentionRollout:
    """
    Attention Rollout importance scores for XSymNet's single attention layer.

    Reference: Abnar & Zuidema, "Quantifying Attention Flow in Transformers" (2020).

    Raw attention weights alone are misleading because the residual connection
    around the attention layer means each position always "attends to itself"
    independently of the learned weights.  Rollout corrects for this by mixing
    the identity matrix (residual path) with the head-averaged attention weights:

        R = 0.5 · I  +  0.5 · A_avg       (I = identity, A_avg = head-averaged weights)
        R ← R / row_sums(R)               (row-normalise)
        importance_j = Σᵢ R[i, j]         (column sum: contribution of position j)

    Resolution note: output length is the feature-map length after the two
    conv+pool stages (~L/8 of the raw input).  Use plot_avg_rollout_with_expt
    to upsample and overlay on the experimental diffraction pattern.

    Args:
        model (nn.Module): Trained XSymNet model (plain or DataParallel-wrapped).
    """

    def __init__(self, model):
        self.model = model
        self._attn_weights: dict = {}
        self._hook_handle = None

    def _get_attention_module(self):
        """Return the MultiheadSelfAttention module, unwrapping DataParallel if needed."""
        m = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        return m.attention

    def _register_hook(self):
        """
        Attach a forward hook to MultiheadSelfAttention that captures the
        attention weight matrix by recomputing Q and K from the module's own
        parameters and the layer input.  This is necessary because
        MultiheadSelfAttention.forward() does not return the weight matrix.
        """
        attn_module = self._get_attention_module()

        def _hook(module, hook_input, _hook_output):
            x = hook_input[0]                                    # (batch, channels, seq_len)
            batch_size, _channels, seq_len = x.shape
            x_perm = x.permute(0, 2, 1)                         # (batch, seq_len, channels)
            with torch.no_grad():
                Q = module.query(x_perm).view(
                    batch_size, seq_len, module.num_heads, module.head_dim
                ).transpose(1, 2)                                # (batch, heads, seq, head_dim)
                K = module.key(x_perm).view(
                    batch_size, seq_len, module.num_heads, module.head_dim
                ).transpose(1, 2)
                scores  = torch.matmul(Q, K.transpose(-2, -1)) / module.scale
                weights = torch.softmax(scores, dim=-1)          # (batch, heads, seq, seq)
                self._attn_weights['A'] = weights.detach().cpu()

        self._hook_handle = attn_module.register_forward_hook(_hook)

    def _remove_hook(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def generate_rollout(self, input_tensor):
        """
        Compute attention rollout importance for a single input.

        Args:
            input_tensor (Tensor): shape (1, seq_len).

        Returns:
            importance (np.ndarray): shape (feat_seq_len,). Non-negative column-sum
                importance at feature-map resolution (~L/8 of raw input length).
                All values are positive and reflect each feature-map position's
                relative contribution through the attention + residual block.
        """
        self._register_hook()
        self.model.eval()
        with torch.no_grad():
            self.model(input_tensor)
        self._remove_hook()

        A     = self._attn_weights['A']           # (1, heads, seq, seq)
        A_avg = A.mean(dim=1).squeeze(0)          # (seq, seq) — averaged over heads

        seq_len = A_avg.shape[0]
        I = torch.eye(seq_len)                    # identity for the residual path

        # Mix attention with residual (equal weight), then row-normalise
        R = 0.5 * I + 0.5 * A_avg
        R = R / R.sum(dim=-1, keepdim=True)

        # Column sum: total contribution of each feature-map position to all outputs
        importance = R.sum(dim=0).numpy()         # (feat_seq_len,)
        return importance

    def generate_average_rollout(self, input_batch):
        """
        Average attention rollout importance over a batch of samples.

        Args:
            input_batch (Tensor): shape (N, seq_len). All from the same subgroup.

        Returns:
            avg_rollout (np.ndarray): shape (feat_seq_len,). Mean column importance
                at feature-map resolution.
        """
        rollouts = []
        print(f"Generating attention rollout for {input_batch.shape[0]} samples...")
        for i in range(input_batch.shape[0]):
            rollout = self.generate_rollout(input_batch[i:i+1])
            rollouts.append(rollout)

        return np.mean(rollouts, axis=0)          # (feat_seq_len,)


#^ ------------------------------------------------------------------------------------------------------
# * Plotting: Integrated Gradients
#^ ------------------------------------------------------------------------------------------------------

def plot_avg_ig_with_expt(xrd_expt, avg_ig, x_axis, x_units='2θ (degrees)',
                          save_name='avg_ig_plot.png', legend_label='',
                          mode='positive', title=None):
    """
    Overlay Integrated Gradients attribution on the experimental XRD pattern.

    Because IG operates at full input resolution, avg_ig and xrd_expt should
    already have the same length — no upsampling artifacts arise.  A small
    resize is applied as a safeguard if lengths differ.

    Args:
        xrd_expt (np.ndarray): shape (seq_len,). Scaled experimental pattern.
        avg_ig (np.ndarray): shape (seq_len,). From IntegratedGradients.generate_average_ig().
        x_axis (array-like): TOF or 2θ values corresponding to seq_len points.
        x_units (str): x-axis label string.
        save_name (str): Output file path (PNG recommended).
        legend_label (str): Legend entry for the pattern line.
        mode (str):
            'positive' — ReLU(IG) then normalise to [0, 1].  Jet colormap.
                         Shows only features that *support* the classification.
                         Directly comparable to GradCAM visualisations.
            'signed'   — Divide by max(|IG|), retain sign.  Coolwarm colormap.
                         Red = supports classification, Blue = opposes it.
        title (str | None): Optional plot title.
    """
    avg_ig   = np.asarray(avg_ig).squeeze()
    xrd_expt = np.asarray(xrd_expt).squeeze()
    x_axis   = np.asarray(x_axis)

    rcParams.update({
        # "text.usetex": False,
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

    # Guard: resize if lengths differ (IG should match xrd_expt, but be safe)
    if avg_ig.shape[0] != xrd_expt.shape[0]:
        print(f"Resizing IG from {avg_ig.shape[0]} to {xrd_expt.shape[0]}")
        avg_ig = np.asarray(zoom(avg_ig, zoom=xrd_expt.shape[0] / avg_ig.shape[0]))
    if x_axis.shape[0] != xrd_expt.shape[0]:
        x_axis = np.asarray(zoom(x_axis, zoom=xrd_expt.shape[0] / x_axis.shape[0]))

    if mode == 'positive':
        display = np.maximum(avg_ig, 0.0)
        display = (display - display.min()) / (display.max() + 1e-8)
        cmap = 'jet'
        vmin, vmax = 0.0, 1.0
    elif mode == 'signed':
        absmax  = np.abs(avg_ig).max() + 1e-8
        display = avg_ig / absmax               # normalised to [-1, 1]
        cmap = 'coolwarm'
        vmin, vmax = -1.0, 1.0
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'positive' or 'signed'.")

    _, ax = plt.subplots(figsize=(12, 6), dpi=600)
    ax.imshow(
        display[np.newaxis, :],
        extent=(x_axis[0], x_axis[-1], float(xrd_expt.min()), float(xrd_expt.max())),
        cmap=cmap, aspect='auto', alpha=0.8, origin='lower',
        vmin=vmin, vmax=vmax
    )
    ax.plot(x_axis, xrd_expt, color='black', linewidth=1.5, label=legend_label)
    if title:
        ax.set_title(title, fontsize=14)
    ax.set_xlabel(x_units, fontsize=14)
    ax.set_ylabel('Intensity A.U.', fontsize=14)
    ax.legend()
    plt.savefig(save_name, dpi=600, bbox_inches='tight')
    # plt.show()
    plt.close()


#^ ------------------------------------------------------------------------------------------------------
# * Plotting: Attention Rollout
#^ ------------------------------------------------------------------------------------------------------

def plot_avg_rollout_with_expt(xrd_expt, avg_rollout, x_axis, x_units='2θ (degrees)',
                               save_name='avg_rollout_plot.png', legend_label='',
                               title=None):
    """
    Overlay Attention Rollout importance on the experimental XRD pattern.

    Attention rollout is at feature-map resolution (~L/8 of the raw input due
    to the two conv+pool stages).  This function *upsamples the rollout* to the
    full input length via scipy.ndimage.zoom so the experimental pattern is
    displayed at its native resolution (contrast with plot_avg_cam_with_expt_zoom_in
    which downsamples the experimental data instead).

    Note: the upsampled rollout will appear smooth.  This smoothness reflects the
    method's fundamental resolution limit (~L/8), not an artifact of this function.

    Args:
        xrd_expt (np.ndarray): shape (seq_len,). Scaled experimental pattern.
        avg_rollout (np.ndarray): shape (feat_seq_len,). From AttentionRollout.generate_average_rollout().
        x_axis (array-like): TOF or 2θ values of length seq_len.
        x_units (str): x-axis label string.
        save_name (str): Output file path.
        legend_label (str): Legend entry for the pattern line.
        title (str | None): Optional plot title.
    """
    avg_rollout = np.asarray(avg_rollout).squeeze()
    xrd_expt   = np.asarray(xrd_expt).squeeze()
    x_axis     = np.asarray(x_axis)

    rcParams.update({
        "text.usetex": False,
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

    # Upsample rollout to match the experimental pattern length
    if avg_rollout.shape[0] != xrd_expt.shape[0]:
        print(f"Upsampling rollout from length {avg_rollout.shape[0]} to {xrd_expt.shape[0]}")
        avg_rollout = np.asarray(zoom(avg_rollout, zoom=xrd_expt.shape[0] / avg_rollout.shape[0]))

    # Normalise to [0, 1]
    avg_rollout = (avg_rollout - avg_rollout.min()) / (avg_rollout.max() + 1e-8)

    _, ax = plt.subplots(figsize=(12, 6), dpi=600)
    ax.imshow(
        avg_rollout[np.newaxis, :],
        extent=(x_axis[0], x_axis[-1], float(xrd_expt.min()), float(xrd_expt.max())),
        cmap='jet', aspect='auto', alpha=0.8, origin='lower',
        vmin=0.0, vmax=1.0
    )
    ax.plot(x_axis, xrd_expt, color='black', linewidth=1.5, label=legend_label)
    if title:
        ax.set_title(title, fontsize=14)
    ax.set_xlabel(x_units, fontsize=14)
    ax.set_ylabel('Intensity A.U.', fontsize=14)
    ax.legend()
    plt.savefig(save_name, dpi=600, bbox_inches='tight')
    # plt.show()
    plt.close()


#^ ------------------------------------------------------------------------------------------------------
# * Previous version of Multi-Head Attention and CNN_MultiHeadAttention (with batchnorm instead of group and layernorm)
#^ ------------------------------------------------------------------------------------------------------

# class MultiheadSelfAttention(nn.Module):
#     def __init__(self, embed_dim, num_heads=4, attn_dropout=0.1):
#         super().__init__()
#         assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by num_heads"
#         self.num_heads = num_heads
#         self.head_dim = embed_dim // num_heads

#         self.query = nn.Linear(embed_dim, embed_dim)
#         self.key   = nn.Linear(embed_dim, embed_dim)
#         self.value = nn.Linear(embed_dim, embed_dim)
#         self.output_proj = nn.Linear(embed_dim, embed_dim)

#         self.scale = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

#     def forward(self, x):
#         batch_size, channels, seq_len = x.shape
#         x = x.permute(0, 2, 1)  # (batch, seq_len, channels)

#         Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
#         K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
#         V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

#         attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
#         attn_weights = torch.softmax(attn_scores, dim=-1)

#         attn_output = torch.matmul(attn_weights, V)
#         attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
#         attn_output = self.output_proj(attn_output)

#         return attn_output.permute(0, 2, 1)


# class CNN_MultiHeadAttention(nn.Module):
#     def __init__(self, num_classes, channel_1, channel_2, kernel_1, kernel_2, stride_1, stride_2, num_heads, linear_1):
#         super().__init__()
        
#         # --- Make ALL input variables instance attributes ---
#         self.num_classes = num_classes
#         self.channel_1   = channel_1
#         self.channel_2   = channel_2
#         self.kernel_1    = kernel_1
#         self.kernel_2    = kernel_2
#         self.stride_1    = stride_1
#         self.stride_2    = stride_2
#         self.num_heads   = num_heads
#         self.linear_1    = linear_1

#         self.conv1 = nn.Conv1d(1, channel_1, kernel_size=kernel_1, stride=stride_1, padding=kernel_1 // 2, bias=False)
#         self.bn1   = nn.BatchNorm1d(channel_1)
#         self.maxpool1 = nn.MaxPool1d(kernel_size=3, stride=2)

#         self.conv2 = nn.Conv1d(channel_1, channel_2, kernel_size=kernel_2, stride=stride_2, padding=kernel_2 // 2, bias=False)
#         self.bn2   = nn.BatchNorm1d(channel_2)
#         self.maxpool2 = nn.MaxPool1d(kernel_size=3, stride=2)

#         self.attention = MultiheadSelfAttention(embed_dim=channel_2, num_heads=num_heads)

#         self.dropout = nn.Dropout(0.2)
#         self.fc1 = nn.LazyLinear(linear_1)
#         self.fc2 = nn.Linear(linear_1, num_classes)

#     def forward(self, x):
#         x = x.unsqueeze(1)
        
#         # x = F.relu(self.bn1(self.conv1(x)))
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = F.relu(x)
#         x = self.maxpool1(x)
        
#         # x = F.relu(self.bn2(self.conv2(x)))
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = F.relu(x)       
#         x = self.maxpool2(x)
        
#         # Multi-head self-attention with residual connection
#         attn_x = self.attention(x)
#         x = x + attn_x

#         x = torch.flatten(x, 1)
#         x = self.dropout(x)
#         x = F.relu(self.fc1(x))
#         return self.fc2(x)