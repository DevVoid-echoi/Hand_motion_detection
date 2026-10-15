"""
numpy_cnn.py - Cài đặt lại HandGestureCNN (model.py) bằng NumPy thuần

Gồm forward (Conv → BatchNorm → ReLU → MaxPool x3 → FC 8192 → 512 → 128 → 9),
backward cho từng lớp và toàn mạng. Đã kiểm chứng với PyTorch autograd và
trọng số models/best_model.pth.
"""
import numpy as np

#           -----Khởi tạo tham số-----
def init_params(num_classes=9, channels=(3, 32, 64, 128), img_size=64, seed=0):
    """
    Khởi tạo trọng số theo He init: w ~ N(0, 2 / fan_in)
    => Giữ phương sai tín hiệu ổn định khi đi qua ReLU, tránh gradient tắt/bùng nổ

    Trả về:
        params:   dict tham số HỌC ĐƯỢC (cập nhật bằng gradient)
                  w1..w3, b1..b3       : Conv
                  gamma1..3, beta1..3  : BatchNorm (scale, shift)
                  w4..w6, b4..b6       : Linear
        bn_state: dict trạng thái BatchNorm KHÔNG học bằng gradient
                  (running_mean, running_var dùng khi eval)
    """
    rng = np.random.default_rng(seed)
    params, bn_state = {}, {}

    # 3 khối conv: kernel 3x3
    for i in (1, 2, 3):
        c_in, c_out = channels[i - 1], channels[i]
        fan_in = c_in * 3 * 3
        params[f"w{i}"] = rng.standard_normal((c_out, c_in, 3, 3)) * np.sqrt(2 / fan_in)
        params[f"b{i}"] = np.zeros(c_out)
        params[f"gamma{i}"] = np.ones(c_out)                # Ban đầu BN không scale
        params[f"beta{i}"] = np.zeros(c_out)                # Ban đầu BN không shift
        bn_state[i] = {"running_mean": np.zeros(c_out), "running_var": np.ones(c_out)}

    # 3 lớp fully connected: 8192 → 512 → 128 → num_classes
    feat = channels[3] * (img_size // 8) ** 2              # 3 lần maxpool → chia 8
    dims = (feat, 512, 128, num_classes)
    for i, (d_in, d_out) in zip((4, 5, 6), zip(dims[:-1], dims[1:])):
        params[f"w{i}"] = rng.standard_normal((d_out, d_in)) * np.sqrt(2 / d_in)
        params[f"b{i}"] = np.zeros(d_out)

    return params, bn_state

#           -----Forward pass-----
def conv_single(x, w, b, stride=1, padding=1):
    """
    Tích chập cho MỘT ảnh

    c: Số kênh màu (R,G,B = 3)
    H = W: Chiều cao và chiều rộng (h = w = 64)
    stride: Mỗi lần trượt bao nhiêu ô
    k: Kích thước kernel
    padding: Thêm bao nhiêu viền 0 để giữ nguyên kích thước

    x: (C_in, H, W)         - Ảnh vào
    w: (C_out, C_in, K, K)  - C_out bộ lọc
    b: (C_out,)             - Bias cho mỗi bộ lọc
    """
    c_in, H, W = x.shape
    c_out, _, k, _ = w.shape

    # Thêm padding
    x_pad = np.pad(x, ((0, 0), (padding, padding), (padding, padding)))

    # Tính kích thước đầu ra: n_out = (n_in + 2*padding - k) // stride + 1
    H_out = (H + 2 * padding - k) // stride + 1
    W_out = (W + 2*padding - k) // stride + 1
    out = np.zeros((c_out, H_out, W_out))

    # Trượt kernel
    for co in range(c_out): # Với từng bộ lọc
        for i in range (H_out): # Với từng hàng
            for j in range (W_out): # Với từng cột
                r, c = i * stride, j * stride
                region = x_pad[:, r: r + k, c: c + k]
                out[co, i, j] = np.sum(region * w[co]) + b[co]

    return out

def conv(x, w, b, stride=1, padding=1):
    """
    Tích chập cho cả batch: chạy conv_single cho từng ảnh rồi xếp lại

    x: (N, C_in, H, W) → out: (N, C_out, H_out, W_out)
    """
    return np.stack([conv_single(xi, w, b, stride, padding) for xi in x])

def batchnorm(x, gamma, beta, bn, train=True, momentum=0.1, eps=1e-5):
    """
    Chuẩn hóa từng kênh về mean ≈ 0, var ≈ 1 rồi scale (gamma) và shift (beta)
    => Phân phối đầu vào các lớp sau ổn định, mạng học nhanh và ổn định hơn

    x: (N, C, H, W)
    gamma, beta: (C,)  - Tham số học được, mỗi kênh 1 cặp
    bn: dict {"running_mean": (C,), "running_var": (C,)} - Cập nhật tại chỗ khi train
    train: True  → dùng mean/var của batch hiện tại, cập nhật running stats
           False → dùng running stats (lúc predict có thể chỉ có 1 ảnh)
    momentum: Tỉ lệ trộn batch stats mới vào running stats
    eps: Tránh chia cho 0 khi var = 0
    """
    N, C, H, W = x.shape
    if train:
        # Mỗi kênh tính trên N ảnh x H x W vị trí → vector (C,)
        mu = x.mean(axis=(0,2,3))
        var = x.var(axis=(0, 2, 3))                     # Chia cho m (biased) để chuẩn hóa
        m = N*H*W
        bn["running_mean"] = (1 - momentum) * bn["running_mean"] + momentum * mu
        # Running var dùng phương sai không chệch (chia m-1) giống PyTorch
        bn["running_var"] = (1 - momentum) * bn["running_var"] + momentum * var * m / (m - 1)
    else:
        mu, var = bn["running_mean"], bn["running_var"]

    # [None, :, None, None]: biến (C,) thành (1, C, 1, 1) để broadcast lên (N, C, H, W)
    std = np.sqrt(var + eps)
    x_hat = (x-mu[None, :, None, None]) / std[None, :, None, None]
    out = gamma[None, :, None, None] * x_hat + beta[None, :, None, None]

    cache = (x_hat, std, gamma)                         # batchnorm_backward cần 3 giá trị này
    return out, cache

def relu(x):
    """
    Hàm kích hoạt, giá trị nhỏ nhất của từng số trong ma trận là 0
    => Tạo nên tính phi tuyến cho phép chồng các lớp tuyến tính lên nhau
    """
    return np.maximum(0, x)

def maxpool(x, k=2):
    """
    Chia bản đồ thành các ô 2x2 và giữ giá trị lớn nhất của mỗi ô
    => Kích thước giảm 1 nửa, lượng tính toán giảm, mạng chịu được sự xê dịch

    x: (N, C, H, W) → (N, C, H // k, W // k)
    """
    n, c, h, w = x.shape
    x = x[:, :, :h // k * k, :w // k * k]               # Cắt bỏ hàng/cột lẻ (nếu có) ở trục H, W
    # Tách H thành (H//k, k) và W thành (W//k, k), lấy max trên 2 trục k
    return x.reshape(n, c, h // k, k, w // k, k).max(axis=(3,5))

def flatten(x):
    """
    Trải thẳng feature map của mỗi ảnh thành vector 1d, giữ nguyên chiều batch

    x: (N, C, H, W) → (N, C*H*W)
    """
    return x.reshape(x.shape[0], -1)

def linear(x, W, b):
    """
    Fully connected
    Nhân với ma trận trọng số và cộng bias
    (Ma trận trọng số W sẽ cập nhật sau mỗi lần học bằng backpropagation)

    x: (N, in), W: (out, in), b: (out,) → (N, out)
    """
    return x @ W.T + b

def dropout(x, p, train=True):
    """
    Inverted dropout: khi train, tắt ngẫu nhiên mỗi neuron với xác suất p
    => Mạng không dựa dẫm vào vài neuron cụ thể, giảm overfitting

    Chia cho (1 - p) để kỳ vọng đầu ra không đổi → lúc eval trả về x nguyên vẹn
    Trả về mask để dropout_backward dùng: dx = dout * mask
    """
    if not train or p==0:
        return x, None
    mask = (np.random.rand(*x.shape) >= p) / (1-p)
    return x * mask, mask

def softmax(z):
    """
    Tính xác suất có tổng = 1 cho từng ảnh
    z: (N, num_classes) - Điểm số sau khi fully connected
    e^z_i / sum(e^z-j) j=0->k
    Trừ max mỗi hàng để e^z không bị tràn số (kết quả không đổi)
    """
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def cross_entropy(probs, y):
    """
    Tính loss, nếu gần đúng thì cross_entropy trả lại nhỏ, nếu sai thì cross_entropy trả lại lớn
    -log(p_y), lấy trung bình trên batch

    probs: (N, num_classes)
    y: (N,) - Nhãn đúng (số nguyên)
    """
    N = probs.shape[0]
    return -np.log(probs[np.arange(N), y] + 1e-12).mean()

def forward_with_cache(x, params, bn_state, train=True):
    """
    Lan truyền xuôi qua toàn mạng, đồng thời lưu lại (cache) các giá trị trung gian
    => Backward cần các giá trị này để tính gradient

    Kiến trúc (giống model.py, ảnh 3x64x64):
        [Conv → BN → ReLU → MaxPool] x3 : 3x64x64 → 32x32x32 → 64x16x16 → 128x8x8
        Flatten                         : 128x8x8 → 8192
        Linear → ReLU → Dropout(0.5)    : 8192 → 512
        Linear → ReLU → Dropout(0.3)    : 512 → 128
        Linear                          : 128 → 9 (logits)

    x: (N, C_in, H, W)  - Batch ảnh vào
    params: dict tham số học được (xem init_params)
    bn_state: dict running stats của BatchNorm (bị cập nhật khi train=True)
    train: True khi huấn luyện, False khi đánh giá/dự đoán (BN dùng running stats, tắt dropout)

    Trả về:
        logits: (N, num_classes) - Điểm số thô của từng lớp (chưa qua softmax)
        cache: dict các giá trị trung gian
    """
    cache = {}

    # 3 khối tích chập: Conv → BN → ReLU → MaxPool
    for i in (1, 2, 3):
        cache[f"x{i}"] = x                              # Input của conv i  → conv_backward cần để tính dw
        z = conv(x, params[f"w{i}"], params[f"b{i}"])
        cache[f"z{i}"] = z                              # Input của BN
        zn, cache[f"bn{i}"] = batchnorm(z, params[f"gamma{i}"], params[f"beta{i}"], bn_state[i], train)

        cache[f"zn{i}"] = zn                            # Input của ReLU    → relu_backward cần biết chỗ nào > 0
        a = relu(zn)
        cache[f"a{i}"] = a                              # Input của MaxPool → maxpool_backward cần biết vị trí max
        x = maxpool(a)                                  # Kích thước giảm 1 nửa

    # Duỗi feature map thành vector
    cache["pool_shape"] = x.shape                       # flatten_backward cần để reshape ngược lại
    f = flatten(x)
    cache["f"] = f                                      # Input của linear 4 → tính dW4

    # Lớp fully connected ẩn 1
    z4 = linear(f, params["w4"], params["b4"])
    cache["z4"] = z4                                    # Input của ReLU     → relu_backward
    a4, cache["mask4"] = dropout(relu(z4), 0.5, train)  # mask4 → dropout_backward
    cache["a4"] = a4                                    # Input của linear 5 → tính dW5

    # Lớp fully connected ẩn 2
    z5 = linear(a4, params["w5"], params["b5"])
    cache["z5"] = z5                                    # Input của ReLU     → relu_backward
    a5, cache["mask5"] = dropout(relu(z5), 0.3, train)  # mask5 → dropout_backward
    cache["a5"] = a5                                    # Input của linear 6 → tính dW6

    # Lớp đầu ra: điểm số cho từng lớp (softmax tính riêng ở ngoài)
    logits = linear(a5, params["w6"], params["b6"])
    return logits, cache

#           -----Backward pass-----
# Quy ước: mỗi hàm nhận dout = dL/d(output của lớp) và trả về
#   dx = dL/d(input của lớp) để truyền tiếp cho lớp phía trước
#   dW, db, ... = gradient tham số của lớp (nếu có)
# Gradient luôn cùng shape với biến tương ứng (dW.shape == W.shape, dx.shape == x.shape)

def softmax_ce_backward(probs, y):
    """
    Gộp softmax + cross-entropy: dL/dz = (p - one_hot(y)) / N
    Lớp đúng nhận gradient âm (logit sẽ tăng), lớp sai nhận gradient dương (logit sẽ giảm)
    Chia N vì loss là trung bình trên batch

    probs: (N, num_classes), y: (N,)
    """
    N = probs.shape[0]
    dz = probs.copy()
    dz[np.arange(N), y] -= 1                            # Trừ 1 tại lớp đúng của từng ảnh
    return dz / N

def linear_backward(dout, x, W):
    """
    Forward: out = x @ W.T + b

    dout: (N, out), x: (N, in) - input lúc forward, W: (out, in)
    """
    dx = dout @ W                                       # (N, in)
    dw = dout.T @ x                                     # (out, in): phép nhân ma trận tự cộng dồn trên N ảnh
    db = dout.sum(axis=0)                               # (out,): b cộng vào mọi ảnh → tổng theo batch
    return dx, dw, db

def dropout_backward(dout, mask):
    """
    Forward: out = x * mask (mask là hằng số) → dx = dout * mask
    Neuron bị tắt thì gradient = 0, neuron giữ lại thì nhân cùng hệ số 1/(1-p)
    mask = None khi train=False (dropout không làm gì) → gradient đi thẳng qua
    """
    return dout if mask is None else dout * mask

def relu_backward(dout, z):
    """
    Đạo hàm ReLU: 1 nếu z > 0, ngược lại 0 → chỉ cho gradient đi qua chỗ đã "mở" lúc forward
    z: input của ReLU lúc forward
    """
    return dout * (z>0)

def unflatten(df, cache):
    """
    Flatten chỉ sắp xếp lại phần tử, không tính toán → backward chỉ cần reshape về shape cũ
    df: (N, C*H*W) → (N, C, H, W)
    """
    return df.reshape(cache["pool_shape"])

def maxpool_backward(dout, a, k=2):
    """
    Output mỗi ô kxk chỉ phụ thuộc phần tử lớn nhất
    => Gradient chỉ chảy về đúng vị trí max, các vị trí khác = 0

    dout: (N, C, H//k, W//k)
    a: (N, C, H, W) - input của maxpool lúc forward
    """
    n, c, h, w = a.shape
    dx = np.zeros_like(a)
    # Lưới chỉ số (ảnh, kênh) để xử lý mọi (n, c) cùng lúc bằng fancy indexing
    nn_, cc = np.meshgrid(np.arange(n), np.arange(c), indexing="ij")
    for i in range(h // k):
        for j in range(w // k):
            win = a[:, :, i*k:(i+1)*k, j*k:(j+1)*k].reshape(n, c, k*k)   # Ô kxk, duỗi thành k*k phần tử
            idx = win.argmax(axis=2)                    # Vị trí max trong ô (chỉ chọn 1 nếu bằng nhau)
            r, s = idx // k, idx % k                    # Đổi về (hàng, cột) trong ô
            dx[nn_, cc, i*k + r, j*k + s] += dout[:, :, i, j]
    return dx

def batchnorm_backward(dout, cache):
    """
    Forward: x_hat = (x - mu) / std, out = gamma * x_hat + beta
    mu, std được tính từ chính x → mỗi x ảnh hưởng output theo 3 đường:
    trực tiếp, qua mu, qua std. Cộng 3 đường lại và rút gọn:
        dx = (m * dx_hat - sum(dx_hat) - x_hat * sum(dx_hat * x_hat)) / (m * std)
    (Chỉ đúng cho train=True; khi eval mu, std là hằng số nhưng không chạy backward)

    dout: (N, C, H, W)
    cache: (x_hat, std, gamma) lưu từ batchnorm forward
    """
    x_hat, std, gamma = cache
    N, C, H, W = dout.shape
    m = N * H * W                                       # Số phần tử mỗi kênh

    # Gamma/beta dùng chung cho mọi (n, h, w) của một kênh → cộng dồn
    dgamma = (dout * x_hat).sum(axis=(0, 2, 3))
    dbeta = dout.sum(axis=(0, 2, 3))

    # Đi ngược qua phép scale gamma
    dx_hat = dout * gamma[None, :, None, None]

    # Đi ngược qua chuẩn hóa
    s1 = dx_hat.sum(axis=(0, 2, 3))[None, :, None, None]                # Đường qua mu: bỏ thành phần dịch đều cả kênh
    s2 = (dx_hat * x_hat).sum(axis=(0, 2, 3))[None, :, None, None]      # Đường qua std: bỏ thành phần co giãn đều
    dx = (m * dx_hat - s1 - x_hat * s2) / (m * std[None, :, None, None])
    return dx, dgamma, dbeta

def conv_single_backward(dout, x, w, stride=1, padding=1):
    """
    Backward tích chập cho MỘT ảnh
    Vòng lặp giống conv_single, nhưng thay vì "đọc vùng → nhân → cộng"
    thì "nhận gradient → rải ngược về vùng đó"

    dout: (C_out, H_out, W_out)
    x: (C_in, H, W) - input của conv lúc forward
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
                dw[co] += g * x_pad[:, r: r + k, c: c + k]          # Bộ lọc dùng ở mọi vị trí → cộng dồn
                dx_pad[:, r: r + k, c: c + k] += g * w[co]          # Pixel nằm trong nhiều vùng → cộng dồn

    # Bỏ viền padding: viền 0 không phải input thật
    dx = dx_pad[:, padding: -padding, padding: -padding] if padding > 0 else dx_pad
    return dx, dw

def conv_backward(dout, x, w, stride=1, padding=1):
    """
    Backward tích chập cho cả batch
    dx tính riêng từng ảnh, dw cộng dồn trên batch (mọi ảnh dùng chung w)

    dout: (N, C_out, H_out, W_out), x: (N, C_in, H, W)
    """
    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    for i in range(x.shape[0]):
        dx[i], dw_i = conv_single_backward(dout[i], x[i], w, stride, padding)
        dw += dw_i
    db = dout.sum(axis=(0, 2, 3))                       # b[co] cộng vào mọi (n, i, j) của kênh co
    return dx, dw, db

def backward(probs, y, cache, params):
    """
    Lan truyền ngược qua toàn mạng, thứ tự ngược với forward_with_cache
    Mỗi lớp lấy input của chính nó lúc forward từ cache

    probs: (N, num_classes) - softmax(logits)
    y: (N,) - Nhãn đúng
    cache: dict từ forward_with_cache (phải chạy với train=True)
    params: dict tham số (cần W để tính dx)

    Trả về:
        g: dict gradient, cùng key và shape với params
           (bn_state không có gradient, đã tự cập nhật trong forward)
    """
    g = {}

    # Lớp đầu ra: softmax + CE → linear 6
    dz6 = softmax_ce_backward(probs, y)
    da5, g["w6"], g["b6"] = linear_backward(dz6, cache["a5"], params["w6"])

    # FC ẩn 2: forward là linear → relu → dropout, nên ngược lại là dropout → relu → linear
    dz5 = relu_backward(dropout_backward(da5, cache["mask5"]), cache["z5"])
    da4, g["w5"], g["b5"] = linear_backward(dz5, cache["a4"], params["w5"])

    # FC ẩn 1
    dz4 = relu_backward(dropout_backward(da4, cache["mask4"]), cache["z4"])
    df, g["w4"], g["b4"] = linear_backward(dz4, cache["f"], params["w4"])

    # (N, 8192) → (N, 128, 8, 8)
    dx = unflatten(df, cache)

    # 3 khối conv, đi ngược 3 → 2 → 1: MaxPool → ReLU → BN → Conv
    # Lưu ý: g["b1..b3"] ≈ 0 vì BN trừ mean đã triệt tiêu bias của conv (không phải lỗi)
    for i in (3, 2, 1):
        da = maxpool_backward(dx, cache[f"a{i}"])
        dzn = relu_backward(da, cache[f"zn{i}"])
        dz, g[f"gamma{i}"], g[f"beta{i}"] = batchnorm_backward(dzn, cache[f"bn{i}"])
        dx, g[f"w{i}"], g[f"b{i}"] = conv_backward(dz, cache[f"x{i}"], params[f"w{i}"])

    return g


