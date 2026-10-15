"""
numpy_cnn.py - Re-implementation of HandGestureCNN (model.py) in pure NumPy

Includes the forward pass (Conv → BatchNorm → ReLU → MaxPool x3 → FC 8192 → 512 → 128 → 9),
backward for each layer and for the whole network. Verified against PyTorch autograd and
the weights in models/best_model.pth.
"""
import numpy as np

#           -----Parameter initialization-----
def init_params(num_classes=9, channels=(3, 32, 64, 128), img_size=64, seed=0):
    """
    Initialize weights with He init: w ~ N(0, 2 / fan_in)
    => Keeps signal variance stable through ReLU, avoiding vanishing/exploding gradients

    Returns:
        params:   dict of LEARNABLE parameters (updated by gradients)
                  w1..w3, b1..b3       : Conv
                  gamma1..3, beta1..3  : BatchNorm (scale, shift)
                  w4..w6, b4..b6       : Linear
        bn_state: dict of BatchNorm state NOT learned by gradients
                  (running_mean, running_var used in eval)
    """
    rng = np.random.default_rng(seed)
    params, bn_state = {}, {}

    # 3 conv blocks: 3x3 kernels
    for i in (1, 2, 3):
        c_in, c_out = channels[i - 1], channels[i]
        fan_in = c_in * 3 * 3
        params[f"w{i}"] = rng.standard_normal((c_out, c_in, 3, 3)) * np.sqrt(2 / fan_in)
        params[f"b{i}"] = np.zeros(c_out)
        params[f"gamma{i}"] = np.ones(c_out)                # BN starts with no scaling
        params[f"beta{i}"] = np.zeros(c_out)                # BN starts with no shift
        bn_state[i] = {"running_mean": np.zeros(c_out), "running_var": np.ones(c_out)}

    # 3 fully connected layers: 8192 → 512 → 128 → num_classes
    feat = channels[3] * (img_size // 8) ** 2              # 3 maxpools → divide by 8
    dims = (feat, 512, 128, num_classes)
    for i, (d_in, d_out) in zip((4, 5, 6), zip(dims[:-1], dims[1:])):
        params[f"w{i}"] = rng.standard_normal((d_out, d_in)) * np.sqrt(2 / d_in)
        params[f"b{i}"] = np.zeros(d_out)

    return params, bn_state

#           -----Forward pass-----
def conv_single(x, w, b, stride=1, padding=1):
    """
    Convolution for ONE image

    c: Number of color channels (R,G,B = 3)
    H = W: Height and width (h = w = 64)
    stride: How many cells to slide each step
    k: Kernel size
    padding: How many zero borders to add to keep the size unchanged

    x: (C_in, H, W)         - Input image
    w: (C_out, C_in, K, K)  - C_out filters
    b: (C_out,)             - Bias for each filter
    """
    c_in, H, W = x.shape
    c_out, _, k, _ = w.shape

    # Add padding
    x_pad = np.pad(x, ((0, 0), (padding, padding), (padding, padding)))

    # Output size: n_out = (n_in + 2*padding - k) // stride + 1
    H_out = (H + 2 * padding - k) // stride + 1
    W_out = (W + 2*padding - k) // stride + 1
    out = np.zeros((c_out, H_out, W_out))

    # Slide the kernel
    for co in range(c_out): # For each filter
        for i in range (H_out): # For each row
            for j in range (W_out): # For each column
                r, c = i * stride, j * stride
                region = x_pad[:, r: r + k, c: c + k]
                out[co, i, j] = np.sum(region * w[co]) + b[co]

    return out

def conv(x, w, b, stride=1, padding=1):
    """
    Convolution for a whole batch: run conv_single on each image and stack the results

    x: (N, C_in, H, W) → out: (N, C_out, H_out, W_out)
    """
    return np.stack([conv_single(xi, w, b, stride, padding) for xi in x])

def batchnorm(x, gamma, beta, bn, train=True, momentum=0.1, eps=1e-5):
    """
    Normalize each channel to mean ≈ 0, var ≈ 1, then scale (gamma) and shift (beta)
    => Inputs to later layers have a stable distribution, so the network learns faster and more stably

    x: (N, C, H, W)
    gamma, beta: (C,)  - Learnable parameters, one pair per channel
    bn: dict {"running_mean": (C,), "running_var": (C,)} - Updated in place during training
    train: True  → use the current batch mean/var and update running stats
           False → use running stats (prediction may have only 1 image)
    momentum: Fraction of new batch stats mixed into running stats
    eps: Avoids division by zero when var = 0
    """
    N, C, H, W = x.shape
    if train:
        # Each channel is computed over N images x H x W positions → vector (C,)
        mu = x.mean(axis=(0,2,3))
        var = x.var(axis=(0, 2, 3))                     # Divide by m (biased) for normalization
        m = N*H*W
        bn["running_mean"] = (1 - momentum) * bn["running_mean"] + momentum * mu
        # Running var uses the unbiased variance (divide by m-1), like PyTorch
        bn["running_var"] = (1 - momentum) * bn["running_var"] + momentum * var * m / (m - 1)
    else:
        mu, var = bn["running_mean"], bn["running_var"]

    # [None, :, None, None]: turn (C,) into (1, C, 1, 1) to broadcast over (N, C, H, W)
    std = np.sqrt(var + eps)
    x_hat = (x-mu[None, :, None, None]) / std[None, :, None, None]
    out = gamma[None, :, None, None] * x_hat + beta[None, :, None, None]

    cache = (x_hat, std, gamma)                         # batchnorm_backward needs these 3 values
    return out, cache

def relu(x):
    """
    Activation function: clamps every value in the matrix to a minimum of 0
    => Adds non-linearity so stacking linear layers is meaningful
    """
    return np.maximum(0, x)

def maxpool(x, k=2):
    """
    Split the map into 2x2 cells and keep the maximum of each cell
    => Halves the size, reduces computation, makes the network tolerant to small shifts

    x: (N, C, H, W) → (N, C, H // k, W // k)
    """
    n, c, h, w = x.shape
    x = x[:, :, :h // k * k, :w // k * k]               # Drop the odd row/column (if any) along H, W
    # Split H into (H//k, k) and W into (W//k, k), take the max over both k axes
    return x.reshape(n, c, h // k, k, w // k, k).max(axis=(3,5))

def flatten(x):
    """
    Flatten each image's feature map into a 1D vector, keeping the batch dimension

    x: (N, C, H, W) → (N, C*H*W)
    """
    return x.reshape(x.shape[0], -1)

def linear(x, W, b):
    """
    Fully connected
    Multiply by the weight matrix and add the bias
    (The weight matrix W is updated after each step by backpropagation)

    x: (N, in), W: (out, in), b: (out,) → (N, out)
    """
    return x @ W.T + b

def dropout(x, p, train=True):
    """
    Inverted dropout: during training, randomly turn off each neuron with probability p
    => The network can't rely on a few specific neurons, reducing overfitting

    Divide by (1 - p) to keep the expected output unchanged → in eval, return x as-is
    Returns the mask for dropout_backward: dx = dout * mask
    """
    if not train or p==0:
        return x, None
    mask = (np.random.rand(*x.shape) >= p) / (1-p)
    return x * mask, mask

def softmax(z):
    """
    Compute probabilities summing to 1 for each image
    z: (N, num_classes) - Scores after the fully connected layer
    e^z_i / sum(e^z-j) j=0->k
    Subtract each row's max so e^z doesn't overflow (result unchanged)
    """
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def cross_entropy(probs, y):
    """
    Compute the loss: small when the prediction is close to correct, large when it is wrong
    -log(p_y), averaged over the batch

    probs: (N, num_classes)
    y: (N,) - True labels (integers)
    """
    N = probs.shape[0]
    return -np.log(probs[np.arange(N), y] + 1e-12).mean()

def forward_with_cache(x, params, bn_state, train=True):
    """
    Forward pass through the whole network, caching intermediate values
    => Backward needs these values to compute gradients

    Architecture (same as model.py, 3x64x64 images):
        [Conv → BN → ReLU → MaxPool] x3 : 3x64x64 → 32x32x32 → 64x16x16 → 128x8x8
        Flatten                         : 128x8x8 → 8192
        Linear → ReLU → Dropout(0.5)    : 8192 → 512
        Linear → ReLU → Dropout(0.3)    : 512 → 128
        Linear                          : 128 → 9 (logits)

    x: (N, C_in, H, W)  - Input image batch
    params: dict of learnable parameters (see init_params)
    bn_state: dict of BatchNorm running stats (updated when train=True)
    train: True for training, False for evaluation/prediction (BN uses running stats, dropout off)

    Returns:
        logits: (N, num_classes) - Raw score per class (before softmax)
        cache: dict of intermediate values
    """
    cache = {}

    # 3 convolution blocks: Conv → BN → ReLU → MaxPool
    for i in (1, 2, 3):
        cache[f"x{i}"] = x                              # Input of conv i   → conv_backward needs it for dw
        z = conv(x, params[f"w{i}"], params[f"b{i}"])
        cache[f"z{i}"] = z                              # Input of BN
        zn, cache[f"bn{i}"] = batchnorm(z, params[f"gamma{i}"], params[f"beta{i}"], bn_state[i], train)

        cache[f"zn{i}"] = zn                            # Input of ReLU    → relu_backward needs to know where > 0
        a = relu(zn)
        cache[f"a{i}"] = a                              # Input of MaxPool → maxpool_backward needs the max positions
        x = maxpool(a)                                  # Size is halved

    # Flatten the feature map into a vector
    cache["pool_shape"] = x.shape                       # flatten_backward needs it to reshape back
    f = flatten(x)
    cache["f"] = f                                      # Input of linear 4 → compute dW4

    # Hidden fully connected layer 1
    z4 = linear(f, params["w4"], params["b4"])
    cache["z4"] = z4                                    # Input of ReLU     → relu_backward
    a4, cache["mask4"] = dropout(relu(z4), 0.5, train)  # mask4 → dropout_backward
    cache["a4"] = a4                                    # Input of linear 5 → compute dW5

    # Hidden fully connected layer 2
    z5 = linear(a4, params["w5"], params["b5"])
    cache["z5"] = z5                                    # Input of ReLU     → relu_backward
    a5, cache["mask5"] = dropout(relu(z5), 0.3, train)  # mask5 → dropout_backward
    cache["a5"] = a5                                    # Input of linear 6 → compute dW6

    # Output layer: score per class (softmax is computed separately outside)
    logits = linear(a5, params["w6"], params["b6"])
    return logits, cache

#           -----Backward pass-----
# Convention: each function takes dout = dL/d(layer output) and returns
#   dx = dL/d(layer input) to pass on to the previous layer
#   dW, db, ... = gradients of the layer's parameters (if any)
# Gradients always have the same shape as their variable (dW.shape == W.shape, dx.shape == x.shape)

def softmax_ce_backward(probs, y):
    """
    Combined softmax + cross-entropy: dL/dz = (p - one_hot(y)) / N
    The correct class gets a negative gradient (its logit increases), wrong classes get a positive gradient (their logits decrease)
    Divide by N because the loss is averaged over the batch

    probs: (N, num_classes), y: (N,)
    """
    N = probs.shape[0]
    dz = probs.copy()
    dz[np.arange(N), y] -= 1                            # Subtract 1 at each image's correct class
    return dz / N

def linear_backward(dout, x, W):
    """
    Forward: out = x @ W.T + b

    dout: (N, out), x: (N, in) - input during forward, W: (out, in)
    """
    dx = dout @ W                                       # (N, in)
    dw = dout.T @ x                                     # (out, in): the matrix product sums over the N images
    db = dout.sum(axis=0)                               # (out,): b is added to every image → sum over the batch
    return dx, dw, db

def dropout_backward(dout, mask):
    """
    Forward: out = x * mask (mask is constant) → dx = dout * mask
    Dropped neurons get gradient = 0, kept neurons are scaled by the same 1/(1-p) factor
    mask = None when train=False (dropout does nothing) → gradient passes straight through
    """
    return dout if mask is None else dout * mask

def relu_backward(dout, z):
    """
    ReLU derivative: 1 if z > 0, otherwise 0 → only lets gradient through where it was "open" in forward
    z: input of ReLU during forward
    """
    return dout * (z>0)

def unflatten(df, cache):
    """
    Flatten only rearranges elements without computing anything → backward just reshapes to the old shape
    df: (N, C*H*W) → (N, C, H, W)
    """
    return df.reshape(cache["pool_shape"])

def maxpool_backward(dout, a, k=2):
    """
    The output of each kxk cell depends only on its largest element
    => Gradient flows back only to the max position, all other positions = 0

    dout: (N, C, H//k, W//k)
    a: (N, C, H, W) - input of maxpool during forward
    """
    n, c, h, w = a.shape
    dx = np.zeros_like(a)
    # Index grid (image, channel) to handle every (n, c) at once with fancy indexing
    nn_, cc = np.meshgrid(np.arange(n), np.arange(c), indexing="ij")
    for i in range(h // k):
        for j in range(w // k):
            win = a[:, :, i*k:(i+1)*k, j*k:(j+1)*k].reshape(n, c, k*k)   # kxk cell, flattened to k*k elements
            idx = win.argmax(axis=2)                    # Max position in the cell (picks only 1 on ties)
            r, s = idx // k, idx % k                    # Convert to (row, column) within the cell
            dx[nn_, cc, i*k + r, j*k + s] += dout[:, :, i, j]
    return dx

def batchnorm_backward(dout, cache):
    """
    Forward: x_hat = (x - mu) / std, out = gamma * x_hat + beta
    mu and std are computed from x itself → each x affects the output via 3 paths:
    directly, through mu, and through std. Summing the 3 paths and simplifying:
        dx = (m * dx_hat - sum(dx_hat) - x_hat * sum(dx_hat * x_hat)) / (m * std)
    (Only valid for train=True; in eval mu, std are constants but backward isn't run)

    dout: (N, C, H, W)
    cache: (x_hat, std, gamma) saved by the batchnorm forward
    """
    x_hat, std, gamma = cache
    N, C, H, W = dout.shape
    m = N * H * W                                       # Number of elements per channel

    # Gamma/beta are shared across every (n, h, w) of a channel → sum up
    dgamma = (dout * x_hat).sum(axis=(0, 2, 3))
    dbeta = dout.sum(axis=(0, 2, 3))

    # Backprop through the gamma scale
    dx_hat = dout * gamma[None, :, None, None]

    # Backprop through normalization
    s1 = dx_hat.sum(axis=(0, 2, 3))[None, :, None, None]                # Path through mu: remove the uniform shift of the channel
    s2 = (dx_hat * x_hat).sum(axis=(0, 2, 3))[None, :, None, None]      # Path through std: remove the uniform scaling
    dx = (m * dx_hat - s1 - x_hat * s2) / (m * std[None, :, None, None])
    return dx, dgamma, dbeta

def conv_single_backward(dout, x, w, stride=1, padding=1):
    """
    Convolution backward for ONE image
    Same loop as conv_single, but instead of "read region → multiply → sum"
    it does "receive gradient → scatter it back to that region"

    dout: (C_out, H_out, W_out)
    x: (C_in, H, W) - input of conv during forward
    w: (C_out, C_in, K, K)
    """
    c_out, _, k, _ = w.shape
    _, H_out, W_out = dout.shape
    x_pad = np.pad(x, ((0, 0), (padding, padding), (padding, padding)))
    dx_pad = np.zeros_like(x_pad)
    dw = np.zeros_like(w)

    for co in range (c_out):
        for i in range(H_out):
            for j in range(W_out):
                r, c = i * stride, j * stride
                g = dout[co, i, j]
                dw[co] += g * x_pad[:, r: r + k, c: c + k]          # The filter is used at every position → accumulate
                dx_pad[:, r: r + k, c: c + k] += g * w[co]          # A pixel lies in many regions → accumulate

    # Strip the padding border: the zero border isn't real input
    dx = dx_pad[:, padding: -padding, padding: -padding] if padding > 0 else dx_pad
    return dx, dw

def conv_backward(dout, x, w, stride=1, padding=1):
    """
    Convolution backward for a whole batch
    dx is computed per image, dw is summed over the batch (all images share w)

    dout: (N, C_out, H_out, W_out), x: (N, C_in, H, W)
    """
    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    for i in range(x.shape[0]):
        dx[i], dw_i = conv_single_backward(dout[i], x[i], w, stride, padding)
        dw += dw_i
    db = dout.sum(axis=(0, 2, 3))                       # b[co] is added to every (n, i, j) of channel co
    return dx, dw, db

def backward(probs, y, cache, params):
    """
    Backpropagation through the whole network, in reverse order of forward_with_cache
    Each layer takes its own forward input from the cache

    probs: (N, num_classes) - softmax(logits)
    y: (N,) - True labels
    cache: dict from forward_with_cache (must be run with train=True)
    params: dict of parameters (W is needed to compute dx)

    Returns:
        g: dict of gradients, same keys and shapes as params
           (bn_state has no gradient, it was already updated in forward)
    """
    g = {}

    # Output layer: softmax + CE → linear 6
    dz6 = softmax_ce_backward(probs, y)
    da5, g["w6"], g["b6"] = linear_backward(dz6, cache["a5"], params["w6"])

    # Hidden FC 2: forward is linear → relu → dropout, so backward is dropout → relu → linear
    dz5 = relu_backward(dropout_backward(da5, cache["mask5"]), cache["z5"])
    da4, g["w5"], g["b5"] = linear_backward(dz5, cache["a4"], params["w5"])

    # Hidden FC 1
    dz4 = relu_backward(dropout_backward(da4, cache["mask4"]), cache["z4"])
    df, g["w4"], g["b4"] = linear_backward(dz4, cache["f"], params["w4"])

    # (N, 8192) → (N, 128, 8, 8)
    dx = unflatten(df, cache)

    # 3 conv blocks, going back 3 → 2 → 1: MaxPool → ReLU → BN → Conv
    # Note: g["b1..b3"] ≈ 0 because BN's mean subtraction cancels the conv bias (not a bug)
    for i in (3, 2, 1):
        da = maxpool_backward(dx, cache[f"a{i}"])
        dzn = relu_backward(da, cache[f"zn{i}"])
        dz, g[f"gamma{i}"], g[f"beta{i}"] = batchnorm_backward(dzn, cache[f"bn{i}"])
        dx, g[f"w{i}"], g[f"b{i}"] = conv_backward(dz, cache[f"x{i}"], params[f"w{i}"])

    return g


