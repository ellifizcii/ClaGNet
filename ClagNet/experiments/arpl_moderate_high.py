from pathlib import Path
# -*- coding: utf-8 -*-
"""
ARPL baseline adapted to scRNA-seq open-set cell annotation.

Reference implementation followed for the ARPL loss:
    Chen et al., "Adversarial Reciprocal Points Learning for Open Set Recognition"
    Official PyTorch repository: https://github.com/gary23ai/ARPL

IMPORTANT:
- This script implements the official ARPLoss / reciprocal-point formulation.
- It does NOT implement the optional ARPL+CS GAN/confusing-sample extension.
- The original ARPL benchmark evaluates unknown detection primarily with max-logit
  scores (AUROC/OSCR) and does not define a universal hard rejection threshold for
  Ao/AKS/AUS/NA. Because this scRNA-seq study requires hard open-set predictions,
  a KNOWN-ONLY validation quantile is used for the hard rejection threshold.
  Unknown cells are NEVER used for threshold selection.
- Preprocessing, known/open splits, batch size, max epochs, and metrics are aligned
  with the user's scRNA-seq comparison protocol.

Outputs:
    arpl_all_runs.csv
    arpl_summary.csv
"""

import os
import random
import warnings
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score

warnings.filterwarnings("ignore")

UNKNOWN = -1
METRICS = ["Ao", "AKS", "AUS", "NA", "FAR", "FRR", "F1_macro", "AUROC"]

# =============================================================================
# SETTINGS
# =============================================================================

DATA_DIR = str(Path(__file__).resolve().parents[1] / "data")
OUTPUT_DIR = "./arpl_openness_020_030_results"

N_RUNS = 30
EPOCHS = 200
PATIENCE = 10

BASE_SEED = 42
N_TOP_GENES = 2000
TEST_SIZE = 0.20
VAL_SIZE = 0.15
UNKNOWN_SAMPLING = "balanced"

HIDDEN_DIM = 256
LATENT_DIM = 64
DROPOUT = 0.25
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4

# Official ARPLoss defaults in the public implementation:
ARPL_TEMP = 1.0
ARPL_WEIGHT_PL = 0.1
ARPL_MARGIN = 1.0
NUM_CENTERS = 1

# Needed only because this paper reports hard open-set metrics.
# Original ARPL uses max reciprocal-point logits as an OOD score; it does not
# prescribe one universal hard threshold for Ao/AKS/AUS/NA.
KNOWN_REJECT_RATE = 0.05

DEVICE = "auto"

DATASETS = [
    {"name": "Li",            "matrix": "data_li.csv",            "labels": "labs_li.csv",            "n_total": 9,  "n_known": 6},
    {"name": "Zeisel",        "matrix": "data_zeisel.csv",        "labels": "labs_zeisel.csv",        "n_total": 9,  "n_known": 6},
    {"name": "Tasic",         "matrix": "data_tasic.csv",         "labels": "labs_tasic.csv",         "n_total": 18, "n_known": 12},
    {"name": "Darmanis",      "matrix": "data_darmanis.csv",      "labels": "labs_darmanis.csv",      "n_total": 9,  "n_known": 6},
    {"name": "Xin",           "matrix": "data_xin.csv",           "labels": "labs_xin.csv",           "n_total": 8,  "n_known": 5},
    {"name": "Kolodziejczyk", "matrix": "data_kolodziejczyk.csv", "labels": "labs_kolodziejczyk.csv", "n_total": 3,  "n_known": 2},
    {"name": "Baron Mouse",   "matrix": "data_baronm1.csv",       "labels": "labs_baronm1.csv",       "n_total": 13, "n_known": 9},
    {"name": "Pollen",        "matrix": "data_pollen.csv",        "labels": "labs_pollen.csv",        "n_total": 11, "n_known": 7},
]


TARGET_OPENNESS_LEVELS = [0.20, 0.30]
N_KNOWN_BY_OPENNESS = {
    "Li": {0.20: 4, 0.30: 3},
    "Zeisel": {0.20: 4, 0.30: 3},
    "Tasic": {0.20: 8, 0.30: 6},
    "Darmanis": {0.20: 4, 0.30: 3},
    "Xin": {0.20: 4, 0.30: 3},
    "Kolodziejczyk": {},
    "Baron Mouse": {0.20: 6, 0.30: 4},
    "Pollen": {0.20: 5, 0.30: 4},
}

def realized_openness(n_known, n_total):
    return 1.0 - np.sqrt(
        (2.0 * float(n_known)) / (float(n_known) + float(n_total))
    )


# =============================================================================
# UTILITIES
# =============================================================================

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_scrna(matrix_path, labels_path, name):
    mat = pd.read_csv(matrix_path)

    genes = mat.iloc[:, 0].astype(str)
    expr = mat.iloc[:, 1:].copy()
    expr.index = genes
    expr = expr.T
    expr.index = pd.Index(range(1, len(expr) + 1), name="cell_id")

    lab = pd.read_csv(labels_path)
    label_col = "cell_type" if "cell_type" in lab.columns else "x"

    if "Unnamed: 0" in lab.columns:
        lab = lab.rename(columns={"Unnamed: 0": "cell_id"})
    elif "cell_id" not in lab.columns:
        lab["cell_id"] = np.arange(1, len(lab) + 1)

    lab["cell_id"] = lab["cell_id"].astype(int)
    lab = lab.set_index("cell_id")

    ids = expr.index.intersection(lab.index)
    expr = expr.loc[ids]
    y_raw = lab.loc[ids, label_col].values

    uniq = sorted(np.unique(y_raw), key=lambda x: str(x))
    mapping = {v: i + 1 for i, v in enumerate(uniq)}
    y = np.array([mapping[v] for v in y_raw], dtype=int)

    # Cell-wise library normalization/log1p does not learn cross-cell parameters.
    lib = expr.sum(axis=1).replace(0, 1.0)
    xlog = np.log1p(expr.div(lib, axis=0) * 1e4).astype(np.float32)

    print(f"[{name}] {xlog.shape[0]} cells | {xlog.shape[1]} genes | {len(uniq)} classes")
    return xlog, y


def split_indices(y, known, open_, seed,
                  test_size=0.20, val_size=0.15,
                  unknown_sampling="balanced"):
    ki = np.where(np.isin(y, known))[0]
    ui = np.where(np.isin(y, open_))[0]

    tr, tk = train_test_split(
        ki, test_size=test_size, random_state=seed, stratify=y[ki]
    )

    opt, val = train_test_split(
        tr, test_size=val_size, random_state=seed, stratify=y[tr]
    )

    if unknown_sampling == "all":
        tu = ui
    else:
        rng = np.random.default_rng(seed)
        uc = np.unique(y[ui])
        per = max(1, len(tk) // 2 // max(len(uc), 1))
        arr = []
        for c in uc:
            ii = ui[y[ui] == c]
            arr.append(rng.choice(ii, size=min(per, len(ii)), replace=False))
        tu = np.concatenate(arr)

    return opt, val, tk, tu


def preprocess_train_only(df, opt, val, tk, tu, n_hvg=2000):
    dopt = df.iloc[opt]

    genes = dopt.var(axis=0).nlargest(
        min(n_hvg, dopt.shape[1])
    ).index

    def arr(idx):
        return df.iloc[idx].loc[:, genes].to_numpy(np.float32, copy=False)

    Xo = arr(opt)
    Xv = arr(val)
    Xk = arr(tk)
    Xu = arr(tu)

    scaler = StandardScaler()
    Xo = scaler.fit_transform(Xo).astype(np.float32)
    Xv = scaler.transform(Xv).astype(np.float32)
    Xk = scaler.transform(Xk).astype(np.float32)
    Xu = scaler.transform(Xu).astype(np.float32)

    return Xo, Xv, Xk, Xu, genes


# =============================================================================
# BACKBONE
# =============================================================================

class DS(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


class ResBlock(nn.Module):
    def __init__(self, d, drop):
        super().__init__()
        self.f1 = nn.Linear(d, d)
        self.b1 = nn.BatchNorm1d(d)
        self.f2 = nn.Linear(d, d)
        self.b2 = nn.BatchNorm1d(d)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        r = x
        x = self.drop(F.gelu(self.b1(self.f1(x))))
        x = self.b2(self.f2(x))
        return F.gelu(x + r)


class FeatureNet(nn.Module):
    """
    Same residual-MLP backbone family used in the scRNA-seq comparison.
    ARPL replaces the ordinary linear classifier with learned reciprocal points.
    """
    def __init__(self, input_dim, hidden=256, latent=64, drop=0.25):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(512, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU()
        )

        self.r1 = ResBlock(hidden, drop)
        self.r2 = ResBlock(hidden, drop)

        self.head = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(128, latent)
        )

    def forward(self, x):
        h = self.r2(self.r1(self.stem(x)))
        # Official ARPL operates directly in feature space; do not L2-normalize.
        return self.head(h)


# =============================================================================
# OFFICIAL ARPL LOSS FORMULATION
# =============================================================================

class Dist(nn.Module):
    """
    Adapted directly from the official ARPL Dist module.

    L2:
        (||x||^2 - 2 x c^T + ||c||^2) / feature_dim
    Dot:
        x c^T
    """
    def __init__(self, num_classes, feat_dim, num_centers=1):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.num_centers = num_centers

        self.centers = nn.Parameter(
            0.1 * torch.randn(
                num_classes * num_centers,
                feat_dim
            )
        )

    def forward(self, features, center=None, metric="l2"):
        if center is None:
            center = self.centers

        if metric == "l2":
            f2 = torch.sum(features.pow(2), dim=1, keepdim=True)
            c2 = torch.sum(center.pow(2), dim=1, keepdim=True)
            dist = (
                f2
                - 2.0 * features.matmul(center.t())
                + c2.t()
            )
            dist = dist / float(features.shape[1])
        else:
            dist = features.matmul(center.t())

        dist = torch.reshape(
            dist,
            [-1, self.num_classes, self.num_centers]
        )
        return torch.mean(dist, dim=2)


class ARPLoss(nn.Module):
    """
    ARPLoss from the official implementation, made device-agnostic.

    Official equations in code:
        dist_dot_p = Dist(x, points, metric='dot')
        dist_l2_p  = Dist(x, points)
        logits     = dist_l2_p - dist_dot_p
        CE(logits / temp, labels)

        dis_known = mean((x - point_y)^2)
        margin ranking(radius, dis_known, target=1)
        loss = CE + weight_pl * margin_loss
    """
    def __init__(
        self,
        num_classes,
        feat_dim,
        temp=1.0,
        weight_pl=0.1,
        margin=1.0,
        num_centers=1
    ):
        super().__init__()

        self.temp = float(temp)
        self.weight_pl = float(weight_pl)

        self.Dist = Dist(
            num_classes=num_classes,
            feat_dim=feat_dim,
            num_centers=num_centers
        )

        self.points = self.Dist.centers

        self.radius = nn.Parameter(
            torch.zeros(1)
        )

        self.margin_loss = nn.MarginRankingLoss(
            margin=float(margin)
        )

    def forward(self, x, labels=None):
        dist_dot_p = self.Dist(
            x,
            center=self.points,
            metric="dot"
        )

        dist_l2_p = self.Dist(
            x,
            center=self.points,
            metric="l2"
        )

        logits = (
            dist_l2_p
            - dist_dot_p
        )

        if labels is None:
            return logits, None

        loss = F.cross_entropy(
            logits / self.temp,
            labels
        )

        # NUM_CENTERS=1 in this study, matching official default.
        centers_1 = self.points.view(
            self.Dist.num_classes,
            self.Dist.num_centers,
            self.Dist.feat_dim
        )[:, 0, :]

        center_batch = centers_1[
            labels,
            :
        ]

        dis_known = (
            x - center_batch
        ).pow(2).mean(1)

        target = torch.ones_like(
            dis_known
        )

        # Expand scalar radius to batch size for modern PyTorch compatibility.
        radius_batch = self.radius.expand_as(
            dis_known
        )

        loss_r = self.margin_loss(
            radius_batch,
            dis_known,
            target
        )

        loss = (
            loss
            + self.weight_pl * loss_r
        )

        return logits, loss


def make_loader(X, y, batch_size, shuffle):
    bs = min(batch_size, len(X))

    drop_last = (
        shuffle
        and len(X) > bs
        and len(X) % bs == 1
    )

    return DataLoader(
        DS(X, y),
        batch_size=bs,
        shuffle=shuffle,
        num_workers=0,
        drop_last=drop_last
    )


def train_arpl(
    Xo, yo, Xv, yv,
    n_classes,
    device,
    seed,
    args
):
    seed_all(seed)

    net = FeatureNet(
        Xo.shape[1],
        args.hidden_dim,
        args.latent_dim,
        args.dropout
    ).to(device)

    criterion = ARPLoss(
        num_classes=n_classes,
        feat_dim=args.latent_dim,
        temp=args.arpl_temp,
        weight_pl=args.arpl_weight_pl,
        margin=args.arpl_margin,
        num_centers=args.num_centers
    ).to(device)

    dl = make_loader(
        Xo, yo,
        args.batch_size,
        True
    )

    vl = make_loader(
        Xv, yv,
        min(512, len(Xv)),
        False
    )

    optimizer = torch.optim.AdamW(
        list(net.parameters())
        + list(criterion.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    best = float("inf")
    best_net = None
    best_criterion = None
    best_epoch = 0
    bad = 0

    for ep in range(1, args.epochs + 1):
        net.train()
        criterion.train()

        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)

            feat = net(xb)
            _, loss = criterion(feat, yb)

            loss.backward()
            optimizer.step()

        net.eval()
        criterion.eval()

        total_loss = 0.0
        n = 0

        with torch.no_grad():
            for xb, yb in vl:
                xb = xb.to(device)
                yb = yb.to(device)

                feat = net(xb)
                _, loss = criterion(feat, yb)

                total_loss += loss.item() * len(xb)
                n += len(xb)

        val_loss = total_loss / max(n, 1)

        if val_loss < best - 1e-5:
            best = val_loss
            best_epoch = ep
            bad = 0

            best_net = {
                k: v.detach().cpu().clone()
                for k, v in net.state_dict().items()
            }

            best_criterion = {
                k: v.detach().cpu().clone()
                for k, v in criterion.state_dict().items()
            }
        else:
            bad += 1

        if ep == 1 or ep % 10 == 0:
            print(
                f"      epoch {ep:03d} "
                f"val_ARPL={val_loss:.4f} "
                f"radius={criterion.radius.item():.4f}"
            )

        if bad >= args.patience:
            break

    if best_net is not None:
        net.load_state_dict(best_net)
        criterion.load_state_dict(best_criterion)

    print(
        f"      selected_epoch={best_epoch} "
        f"best_val_ARPL={best:.4f} "
        f"radius={criterion.radius.item():.4f}"
    )

    return net, criterion, best_epoch


@torch.no_grad()
def get_logits(net, criterion, X, device):
    net.eval()
    criterion.eval()

    dl = make_loader(
        X,
        np.zeros(len(X), dtype=int),
        min(512, len(X)),
        False
    )

    all_logits = []

    for xb, _ in dl:
        feat = net(
            xb.to(device)
        )
        logits, _ = criterion(
            feat,
            labels=None
        )
        all_logits.append(
            logits.cpu().numpy()
        )

    return np.vstack(
        all_logits
    )


# =============================================================================
# OPEN-SET DECISION
# =============================================================================

def calibrate_known_threshold(logits_val, reject_rate=0.05):
    """
    ARPL's official evaluation uses max(logits) as the knownness/OOD score.
    To obtain hard unknown predictions for this paper's Ao/AKS/AUS/NA metrics,
    calibrate a lower-tail threshold using KNOWN validation cells only.
    """
    score = np.max(
        logits_val,
        axis=1
    )

    return float(
        np.quantile(
            score,
            reject_rate
        )
    )


def predict_arpl(logits, known, threshold):
    known = np.asarray(
        sorted(known),
        dtype=int
    )

    local_pred = logits.argmax(
        axis=1
    )

    score = logits.max(
        axis=1
    )

    pred = known[
        local_pred
    ].copy()

    pred[
        score < threshold
    ] = UNKNOWN

    return pred, score


# =============================================================================
# METRICS
# =============================================================================

def metrics(y, p, score):
    kn = y != UNKNOWN
    un = ~kn

    yk = y[kn]
    pk = p[kn]
    pu = p[un]

    TP = np.sum(pk == yk)
    FN = np.sum(pk != yk)
    TU = np.sum(pu == UNKNOWN)
    FU = np.sum(pu != UNKNOWN)

    n = len(y)

    Ao = (TP + TU) / n if n else 0.0
    AKS = TP / (TP + FN) if (TP + FN) else 0.0
    AUS = TU / (TU + FU) if (TU + FU) else 0.0
    FAR = FU / (TU + FU) if (TU + FU) else 0.0
    FRR = FN / (TP + FN) if (TP + FN) else 0.0
    NA = 0.5 * (AKS + AUS)

    labs = sorted(
        np.unique(
            y[y != UNKNOWN]
        )
    )

    F1 = f1_score(
        y,
        p,
        labels=labs,
        average="macro",
        zero_division=0
    )

    try:
        AUROC = roc_auc_score(
            (y != UNKNOWN).astype(int),
            score
        )
    except Exception:
        AUROC = np.nan

    return dict(
        Ao=Ao,
        AKS=AKS,
        AUS=AUS,
        NA=NA,
        FAR=FAR,
        FRR=FRR,
        F1_macro=F1,
        AUROC=AUROC
    )


def summarize(df):
    rows = []

    for name, g in df.groupby("Dataset"):
        row = {
            "Dataset": name,
            "Method": "ARPL",
            "N_Runs": len(g)
        }

        for m in METRICS:
            row[m + "_mean"] = g[m].mean()
            row[m + "_std"] = g[m].std(ddof=0)

        rows.append(row)

    return pd.DataFrame(rows)


def print_summary(summary):
    print(
        "\n"
        "                    FINAL SUMMARY — ARPL\n"
        "                      (LEAKAGE-FREE)\n"
    )

    print(
        f"{'Dataset':<20}"
        f"{'Ao':>8}"
        f"{'AKS':>8}"
        f"{'AUS':>8}"
        f"{'NA':>8}"
        f"{'AUROC':>8}"
        f"{'F1':>8}"
    )

    print("-" * 76)

    for _, r in summary.iterrows():
        print(
            f"{r.Dataset:<20}"
            f"{r.Ao_mean:>8.4f}"
            f"{r.AKS_mean:>8.4f}"
            f"{r.AUS_mean:>8.4f}"
            f"{r.NA_mean:>8.4f}"
            f"{r.AUROC_mean:>8.4f}"
            f"{r.F1_macro_mean:>8.4f}"
        )


# =============================================================================
# CONFIG
# =============================================================================

class Args:
    pass


def build_args():
    args = Args()

    args.data_dir = DATA_DIR
    args.output_dir = OUTPUT_DIR

    args.n_runs = N_RUNS
    args.epochs = EPOCHS
    args.patience = PATIENCE

    args.base_seed = BASE_SEED
    args.n_top_genes = N_TOP_GENES
    args.test_size = TEST_SIZE
    args.val_size = VAL_SIZE
    args.unknown_sampling = UNKNOWN_SAMPLING

    args.hidden_dim = HIDDEN_DIM
    args.latent_dim = LATENT_DIM
    args.dropout = DROPOUT
    args.batch_size = BATCH_SIZE
    args.lr = LR
    args.weight_decay = WEIGHT_DECAY

    args.arpl_temp = ARPL_TEMP
    args.arpl_weight_pl = ARPL_WEIGHT_PL
    args.arpl_margin = ARPL_MARGIN
    args.num_centers = NUM_CENTERS

    args.known_reject_rate = KNOWN_REJECT_RATE

    args.device = DEVICE

    return args


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = build_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA istendi ama bu bilgisayarda kullanilamiyor.")

    device = torch.device(
        "cuda" if (args.device == "cuda" or
        (args.device == "auto" and torch.cuda.is_available())) else "cpu"
    )

    print("Device:", device)
    print("Target openness levels:", TARGET_OPENNESS_LEVELS)
    rows = []

    for ds in DATASETS:
        name = ds["name"]
        try:
            Xdf, y = load_scrna(
                os.path.join(args.data_dir, ds["matrix"]),
                os.path.join(args.data_dir, ds["labels"]), name
            )
        except Exception as e:
            print("SKIP", name, type(e).__name__, e)
            continue

        classes = sorted(np.unique(y))
        n_total = len(classes)

        for target_o in TARGET_OPENNESS_LEVELS:
            cfg = N_KNOWN_BY_OPENNESS.get(name, {})
            if target_o not in cfg:
                print(f"\n[{name}] Target O={target_o:.2f} SKIP")
                continue

            n_known = cfg[target_o]
            n_open = n_total - n_known
            if n_known < 2 or n_known >= n_total:
                print(f"\n[{name}] Target O={target_o:.2f} SKIP: invalid K={n_known}")
                continue

            real_o = realized_openness(n_known, n_total)
            print(f"\n{'#'*90}")
            print(f"{name} | Target O={target_o:.2f} | Realized O={real_o:.4f} | {n_known}K/{n_open}U")
            print("#"*90)

            for run in range(args.n_runs):
                seed = args.base_seed + 17 * run
                class_seed = args.base_seed + 13 * run
                rng = np.random.default_rng(class_seed)
                known = sorted(rng.choice(classes, n_known, replace=False).tolist())
                open_ = [c for c in classes if c not in known]

                print(
                    f"\n[{name}] O_target={target_o:.2f} O_real={real_o:.4f} "
                    f"Run {run+1}/{args.n_runs} Known={known} Open={open_}"
                )

                try:
                    io, iv, itk, itu = split_indices(
                        y, known, open_, seed, args.test_size,
                        args.val_size, args.unknown_sampling
                    )
                    Xo, Xv, Xk, Xu, genes = preprocess_train_only(
                        Xdf, io, iv, itk, itu, args.n_top_genes
                    )
                    yo, yv, yk = y[io], y[iv], y[itk]
                    g2l = {c:i for i,c in enumerate(known)}
                    yol = np.array([g2l[c] for c in yo], dtype=int)
                    yvl = np.array([g2l[c] for c in yv], dtype=int)

                    net, criterion, best_epoch = train_arpl(
                        Xo, yol, Xv, yvl, len(known), device, seed, args
                    )

                    Lv = get_logits(net, criterion, Xv, device)
                    threshold = calibrate_known_threshold(Lv, args.known_reject_rate)

                    Lk = get_logits(net, criterion, Xk, device)
                    Lu = get_logits(net, criterion, Xu, device)
                    Ltest = np.vstack([Lk, Lu])
                    ytest = np.r_[yk, np.full(len(Lu), UNKNOWN, dtype=int)]

                    pred, score = predict_arpl(Ltest, known, threshold)
                    met = metrics(ytest, pred, score)

                    rows.append({
                        "Dataset": name, "Method": "ARPL", "Run": run+1,
                        "Seed": seed, "Class_Seed": class_seed,
                        "Best_Epoch": best_epoch,
                        "Target_Openness": target_o,
                        "Realized_Openness": real_o,
                        "N_Known": n_known, "N_Open": n_open, "N_Total": n_total,
                        "Known_Classes": "|".join(map(str, known)),
                        "Open_Classes": "|".join(map(str, open_)),
                        "ARPL_Temp": args.arpl_temp,
                        "ARPL_Weight_PL": args.arpl_weight_pl,
                        "ARPL_Margin": args.arpl_margin,
                        "ARPL_Radius": float(criterion.radius.detach().cpu().item()),
                        "Known_Score_Threshold": threshold,
                        **met
                    })

                    print(
                        f" -> Ao={met['Ao']:.4f} AKS={met['AKS']:.4f} "
                        f"AUS={met['AUS']:.4f} NA={met['NA']:.4f} "
                        f"AUROC={met['AUROC']:.4f} F1={met['F1_macro']:.4f}"
                    )
                    pd.DataFrame(rows).to_csv(
                        os.path.join(args.output_dir, "arpl_openness_020_030_all_runs.csv"),
                        index=False
                    )
                except Exception as e:
                    print(" RUN ERROR:", type(e).__name__, e)

    if not rows:
        print("\nNo successful run was produced.")
        return

    df = pd.DataFrame(rows)
    summary_rows = []
    groups = ["Dataset","Target_Openness","Realized_Openness","N_Known","N_Open","N_Total"]

    for keys, g in df.groupby(groups):
        name, target_o, real_o, nk, nu, nt = keys
        r = {
            "Dataset":name, "Method":"ARPL", "Target_Openness":target_o,
            "Realized_Openness":real_o, "N_Known":nk, "N_Open":nu,
            "N_Total":nt, "N_Runs":len(g)
        }
        for metric in METRICS:
            r[metric+"_mean"] = g[metric].mean()
            r[metric+"_std"] = g[metric].std(ddof=0)
        summary_rows.append(r)

    summary = pd.DataFrame(summary_rows).sort_values(["Target_Openness","Dataset"])
    summary.to_csv(
        os.path.join(args.output_dir, "arpl_openness_020_030_summary.csv"),
        index=False
    )

    for target_o in sorted(summary["Target_Openness"].unique()):
        q = summary[summary["Target_Openness"] == target_o]
        print(f"\nTARGET OPENNESS = {target_o:.2f}")
        print(f"{'Dataset':<20}{'Real.O':>8}{'K/U':>9}{'Ao':>8}{'AKS':>8}{'AUS':>8}{'NA':>8}{'AUROC':>8}{'F1':>8}")
        print("-"*86)
        for _, r in q.iterrows():
            ku=f"{int(r.N_Known)}/{int(r.N_Open)}"
            print(
                f"{r.Dataset:<20}{r.Realized_Openness:>8.3f}{ku:>9}"
                f"{r.Ao_mean:>8.4f}{r.AKS_mean:>8.4f}{r.AUS_mean:>8.4f}"
                f"{r.NA_mean:>8.4f}{r.AUROC_mean:>8.4f}{r.F1_macro_mean:>8.4f}"
            )


if __name__ == "__main__":
    main()
