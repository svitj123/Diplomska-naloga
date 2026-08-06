import h5py
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path





def neighborhood_features(data, rows, cols, k=3):
    pad = k // 2
    h, w, d = data.shape

    feats = np.zeros((len(rows), d * 4), dtype=np.float32)

    for i, (r, c) in enumerate(zip(rows, cols)):
        r0, r1 = max(0, r - pad), min(h, r + pad + 1)
        c0, c1 = max(0, c - pad), min(w, c + pad + 1)

        patch = data[r0:r1, c0:c1]

        mean = patch.mean(axis=(0,1))
        std = patch.std(axis=(0,1))
        maxv = patch.max(axis=(0,1))
        minv = patch.min(axis=(0,1))

        feats[i] = np.concatenate([mean, std, maxv, minv])

    return feats


def stacked_neighborhood_features(data, rows, cols):
    feats_3 = neighborhood_features(data, rows, cols, k=3)
    feats_5 = neighborhood_features(data, rows, cols, k=5)
    return np.concatenate([feats_3, feats_5], axis=1)


def g_kernel(sigma, device):
    r = int(np.ceil(4.0 * sigma))
    coords = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    k = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return k / k.sum(), r


def masked_gaussian_probs(prob, mask, sigma, eps, device):
    if sigma <= 0:
        return prob.astype(np.float32)

    prob = torch.as_tensor(prob, device=device, dtype=torch.float32)
    mask = torch.as_tensor(mask, device=device, dtype=torch.float32)

    H, W, C = prob.shape
    k, r = g_kernel(sigma, device)
    k = k[None, None]
    w = k.repeat(C, 1, 1, 1)

    x = prob.permute(2, 0, 1)[None]   # (1, C, H, W)
    m = mask[None, None]              # (1, 1, H, W)
    x = F.pad(x * m, (r, r, r, r), mode="replicate")
    m = F.pad(m, (r, r, r, r), mode="replicate")
    num = F.conv2d(x, w, groups=C)
    den = F.conv2d(m, k).clamp_min_(1e-8)
    out = (num / den)[0].permute(1, 2, 0)
    out = out.clamp_min(eps)
    out /= out.sum(dim=-1, keepdim=True)
    return out.cpu().numpy().astype(np.float32)

def train_torch_mlp(X, y, l2_strength, max_iter, seed, device):
    torch.manual_seed(seed)

    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y, dtype=torch.long, device=device)

    model = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 6)
    ).to(device)

    
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, std=0.01)
            torch.nn.init.zeros_(m.bias)

    model.apply(init_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    for _ in range(max_iter):
        optimizer.zero_grad()

        logits = model(X_tensor)
        loss = loss_fn(logits, y_tensor)
        loss = loss + 0.5 * l2_strength * sum(
            torch.sum(p * p) for p in model.parameters() if p.requires_grad
        )

        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(X_tensor)
        train_ce = loss_fn(logits, y_tensor)

        train_loss = train_ce.item()
        train_objective = train_loss
        train_acc = (logits.argmax(dim=1) == y_tensor).float().mean().item()

    return model, train_objective, train_loss, train_acc


def get_probas(model, X, batch_size, device):
    probabilities = []
    model.eval()
    with torch.no_grad():
        for start in range(0, X.shape[0], batch_size):
            stop = start + batch_size
            xb = torch.tensor(X[start:stop], dtype=torch.float32, device=device)
            probabilities.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    return np.concatenate(probabilities, axis=0).astype(np.float32)


def main():
    input_path = Path("image1-competition.hdf5")
    output_path = Path("submission.npy")
    l2_strength = 0.03
    sigma = 1.5
    max_iter = 300
    batch_size = 8192
    seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    with h5py.File(input_path, "r") as f:
        data = np.array(f["data"], dtype=np.float32)
        classes = np.array(f["classes"])
        tissue_mask = np.array(f["tissue_mask"], dtype=bool)

    annotated = classes != -1
    X = data[annotated]
    rows, cols = np.where(annotated)

    neigh = stacked_neighborhood_features(data, rows, cols)

    X = np.concatenate([X, neigh], axis=1)
    y = classes[annotated].astype(np.int64)
    
    # standardize
    mean = X.mean(axis=0).astype(np.float32)
    std = X.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    X = (X - mean) / std
    

    
    model, final_loss, train_loss, train_acc = train_torch_mlp(
        X, y, l2_strength, max_iter, seed, device
    )
    print(f"train_loss={train_loss:.6f} train_acc={train_acc:.4f} optimizer_loss={final_loss:.6f}")
    rows_test, cols_test = slice(265, 465), slice(360, 660) 
    n_classes = 6
    crop = data[rows_test, cols_test]
    crop_mask = tissue_mask[rows_test, cols_test]
    X_pred = crop.reshape(-1, crop.shape[-1])
    
    pred_rows = np.repeat(np.arange(rows_test.start, rows_test.stop),
                      crop.shape[1])
    pred_cols = np.tile(np.arange(cols_test.start, cols_test.stop),
                        crop.shape[0])

    neigh_pred = stacked_neighborhood_features(data, pred_rows, pred_cols)

    X_pred = np.concatenate([X_pred, neigh_pred], axis=1)
    X_pred = (X_pred - mean) / std
    prob = get_probas(model, X_pred, batch_size, device)
    prob = prob.reshape(crop.shape[:2] + (n_classes,))
    eps = np.float32(1e-6)
    smoothed = masked_gaussian_probs(prob, crop_mask, sigma, eps=1e-7, device=device)
    final = prob.copy()
    final[crop_mask] = smoothed[crop_mask]
    final = np.clip(final, eps, np.float32(1.0) - eps)
    final /= final.sum(axis=-1, keepdims=True)
    final = final.astype(np.float32)

    output = Path(output_path)
    np.save(output, final)
    print(f"predictions saved to {output.resolve()} with shape {final.shape} and dtype {final.dtype}")
    
    


if __name__ == "__main__":
    main()
