from pathlib import Path
# -*- coding: utf-8 -*-
"""
OpenMax baseline for scRNA-seq open-set cell annotation using LibMR EVT fitting.

Design goal:
- Preserve the SAME dataset handling, known/open class sampling, train/validation/test
  protocol, preprocessing, backbone size, batch size, epochs and metrics as the user's
  Deep Open-Set GMM code as closely as possible.
- Replace the GMM open-set decision mechanism with OpenMax.

Method-faithful OpenMax choices:
- Network is trained with standard cross-entropy only (no SupCon).
- Mean Activation Vectors (MAVs) are estimated from correctly classified optimization
  samples only.
- Per-class EVT tails are fit with LibMR MR.fit_high, as in the original OpenMax code.
- The original eucos distance scaling is used: Euclidean/200 + cosine distance.
- At inference, top-alpha activations are recalibrated and removed activation mass
  is assigned to an explicit unknown class.
- No extra validation-derived rejection rule is added; the explicit OpenMax unknown
  posterior is used for the final rejection decision.
- Unknown data are used only in final test evaluation.

Requires:
    pip install numpy pandas scikit-learn scipy torch

Outputs:
    openmax_all_runs.csv
    openmax_summary.csv
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

try:
    import libmr
except ImportError as e:
    raise ImportError(
        "\nOpenMax'in orijinal EVT fitting mekanizmasi icin 'libmr' gereklidir.\n"
        "Bu surum bilerek SciPy fallback kullanmaz; aksi halde baseline canonical OpenMax'ten sapar.\n"
        "Once kullandiginiz Python/Spyder ortaminda libmr kurulumunu tamamlayin.\n"
    ) from e

warnings.filterwarnings("ignore")

UNKNOWN = -1
METRICS = ["Ao", "AKS", "AUS", "NA", "FAR", "FRR", "F1_macro", "AUROC"]

# =============================================================================
# SETTINGS
# =============================================================================

DATA_DIR = str(Path(__file__).resolve().parents[1] / "data")
OUTPUT_DIR = "./openmax_openness_020_030_results"

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

# OpenMax parameters
TAIL_SIZE = 20
ALPHA_RANK = 3
# Original OpenMax eucos distance uses Euclidean/200 + cosine.
EUCOS_EUCLIDEAN_DIVISOR = 200.0

# Canonical-style OpenMax decision:
# no additional known-validation rejection threshold is used.

DEVICE = "auto"   # "auto", "cpu", "cuda"

DATASETS = [
    {"name": "Li",            "matrix": "data_li.csv",              "labels": "labs_li.csv",              "n_total": 9},
    {"name": "Zeisel",        "matrix": "data_zeisel.csv",          "labels": "labs_zeisel.csv",          "n_total": 9},
    {"name": "Tasic",         "matrix": "data_tasic.csv",           "labels": "labs_tasic.csv",           "n_total": 18},
    {"name": "Darmanis",      "matrix": "data_darmanis.csv",        "labels": "labs_darmanis.csv",        "n_total": 9},
    {"name": "Xin",           "matrix": "data_xin.csv",             "labels": "labs_xin.csv",             "n_total": 8},
    {"name": "Kolodziejczyk", "matrix": "data_kolodziejczyk.csv",   "labels": "labs_kolodziejczyk.csv",   "n_total": 3},
    {"name": "Baron Mouse",   "matrix": "data_baronm1.csv",         "labels": "labs_baronm1.csv",         "n_total": 13},
    {"name": "Pollen",        "matrix": "data_pollen.csv",          "labels": "labs_pollen.csv",          "n_total": 11},
]

TARGET_OPENNESS_LEVELS = [0.20, 0.30]

N_KNOWN_BY_OPENNESS = {
    "Li":            {0.20: 4, 0.30: 3},
    "Zeisel":        {0.20: 4, 0.30: 3},
    "Tasic":         {0.20: 8, 0.30: 6},
    "Darmanis":      {0.20: 4, 0.30: 3},
    "Xin":           {0.20: 4, 0.30: 3},
    "Kolodziejczyk": {},
    "Baron Mouse":   {0.20: 6, 0.30: 4},
    "Pollen":        {0.20: 5, 0.30: 4},
}

def realized_openness(n_known, n_total):
    return 1.0 - np.sqrt(
        (2.0 * float(n_known)) /
        (float(n_known) + float(n_total))
    )



# =============================================================================
# BASIC UTILITIES
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

    # Cell-wise normalization only; safe before known/open split.
    lib = expr.sum(axis=1).replace(0, 1.0)
    xlog = np.log1p(expr.div(lib, axis=0) * 1e4).astype(np.float32)

    print(
        f"[{name}] {xlog.shape[0]} cells | "
        f"{xlog.shape[1]} genes | {len(uniq)} classes"
    )

    return xlog, y


def split_indices(
    y,
    known,
    open_,
    seed,
    test_size=0.20,
    val_size=0.15,
    unknown_sampling="balanced"
):
    ki = np.where(np.isin(y, known))[0]
    ui = np.where(np.isin(y, open_))[0]

    tr, tk = train_test_split(
        ki,
        test_size=test_size,
        random_state=seed,
        stratify=y[ki]
    )

    opt, val = train_test_split(
        tr,
        test_size=val_size,
        random_state=seed,
        stratify=y[tr]
    )

    if unknown_sampling == "all":
        tu = ui
    else:
        rng = np.random.default_rng(seed)
        uc = np.unique(y[ui])

        per = max(
            1,
            len(tk) // 2 // max(len(uc), 1)
        )

        arr = []
        for c in uc:
            ii = ui[y[ui] == c]
            arr.append(
                rng.choice(
                    ii,
                    size=min(per, len(ii)),
                    replace=False
                )
            )

        tu = np.concatenate(arr)

    return opt, val, tk, tu


def preprocess_train_only(
    df,
    opt,
    val,
    tk,
    tu,
    n_hvg=2000
):
    dopt = df.iloc[opt]

    genes = dopt.var(axis=0).nlargest(
        min(n_hvg, dopt.shape[1])
    ).index

    def arr(idx):
        return (
            df.iloc[idx]
            .loc[:, genes]
            .to_numpy(np.float32, copy=False)
        )

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
# NETWORK
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
        x = self.drop(
            F.gelu(
                self.b1(
                    self.f1(x)
                )
            )
        )
        x = self.b2(
            self.f2(x)
        )
        return F.gelu(x + r)


class Net(nn.Module):
    """
    Same backbone family as the user's Deep Open-Set GMM model.

    Important OpenMax detail:
    OpenMax uses the classifier activation vector BEFORE softmax.
    Therefore forward returns:
        z      = latent representation
        logits = activation vector used by OpenMax
    """

    def __init__(
        self,
        input_dim,
        n_classes,
        hidden=256,
        latent=64,
        drop=0.25
    ):
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

        self.cls = nn.Linear(
            latent,
            n_classes
        )

    def forward(self, x):
        h = self.r2(
            self.r1(
                self.stem(x)
            )
        )

        z = F.normalize(
            self.head(h),
            p=2,
            dim=1
        )

        logits = self.cls(z)

        return z, logits


def make_loader(X, y, batch_size, shuffle):
    bs = min(batch_size, len(X))

    # BatchNorm guard: avoid final mini-batch of exactly one sample.
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


def train_model(
    Xo,
    yo,
    Xv,
    yv,
    n_classes,
    device,
    seed,
    args
):
    seed_all(seed)

    model = Net(
        Xo.shape[1],
        n_classes,
        args.hidden_dim,
        args.latent_dim,
        args.dropout
    ).to(device)

    dl = make_loader(
        Xo,
        yo,
        args.batch_size,
        True
    )

    vl = make_loader(
        Xv,
        yv,
        min(512, len(Xv)),
        False
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    ce = nn.CrossEntropyLoss()

    best = float("inf")
    state = None
    bad = 0
    best_epoch = 0

    for ep in range(1, args.epochs + 1):
        model.train()

        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(
                set_to_none=True
            )

            _, logits = model(xb)

            # Canonical OpenMax baseline:
            # standard discriminative cross-entropy training.
            loss = ce(logits, yb)

            loss.backward()
            optimizer.step()

        model.eval()

        s = 0.0
        n = 0

        with torch.no_grad():
            for xb, yb in vl:
                xb = xb.to(device)
                yb = yb.to(device)

                _, logits = model(xb)
                l = ce(logits, yb)

                s += l.item() * len(xb)
                n += len(xb)

        val_ce = s / max(n, 1)

        if val_ce < best - 1e-5:
            best = val_ce
            state = {
                k: t.detach().cpu().clone()
                for k, t in model.state_dict().items()
            }
            bad = 0
            best_epoch = ep
        else:
            bad += 1

        if ep == 1 or ep % 10 == 0:
            print(
                f"      epoch {ep:03d} "
                f"val_CE={val_ce:.4f}"
            )

        if bad >= args.patience:
            break

    if state is not None:
        model.load_state_dict(state)

    print(
        f"      selected_epoch={best_epoch} "
        f"best_val_CE={best:.4f}"
    )

    return model, best_epoch


@torch.no_grad()
def get_activations(model, X, device):
    """
    Returns:
        latent Z
        activation vectors A = logits before softmax
        closed-set predicted local class index
    """
    model.eval()

    z_all = []
    a_all = []

    dl = make_loader(
        X,
        np.zeros(len(X), dtype=int),
        min(512, len(X)),
        False
    )

    for xb, _ in dl:
        z, logits = model(
            xb.to(device)
        )

        z_all.append(
            z.cpu().numpy()
        )

        a_all.append(
            logits.cpu().numpy()
        )

    Z = np.vstack(z_all)
    A = np.vstack(a_all)
    pred_local = A.argmax(axis=1)

    return Z, A, pred_local


# =============================================================================
# OPENMAX
# =============================================================================

def cosine_distance(a, b, eps=1e-12):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)

    if na < eps or nb < eps:
        return 0.0

    return 1.0 - (
        np.dot(a, b)
        / (na * nb)
    )


def eucos_distance(
    a,
    b,
    euclidean_divisor=200.0
):
    """
    OpenMax eucos distance used by the original implementation family:
        Euclidean(a,b) / 200 + cosine_distance(a,b)
    """
    eu = np.linalg.norm(a - b) / euclidean_divisor
    co = cosine_distance(a, b)
    return float(eu + co)


def fit_openmax(
    activations,
    y_global,
    pred_local,
    known,
    tail_size=20,
    euclidean_divisor=200.0
):
    """
    Fit one MAV + LibMR meta-recognition model per known class.

    Following OpenMax:
    - use correctly classified optimization samples for each class;
    - compute the class MAV in activation-vector space;
    - compute eucos distances to the MAV;
    - fit LibMR to the largest tail distances using MR.fit_high.

    A fallback to all class samples is used only if a class has zero correctly
    classified optimization cells, so that small scRNA-seq runs do not crash.
    This fallback is logged in the output.
    """

    known = sorted(known)
    global_to_local = {c: i for i, c in enumerate(known)}
    models = {}

    for c in known:
        local_c = global_to_local[c]

        class_mask = (y_global == c)
        correct_mask = class_mask & (pred_local == local_c)

        A_c = activations[correct_mask]
        fallback_used = False

        if len(A_c) == 0:
            A_c = activations[class_mask]
            fallback_used = True

        if len(A_c) == 0:
            raise RuntimeError(
                f"OpenMax: class {c} has no optimization samples."
            )

        mav = A_c.mean(axis=0).astype(np.float64)

        distances = np.asarray(
            [
                eucos_distance(
                    a,
                    mav,
                    euclidean_divisor
                )
                for a in A_c
            ],
            dtype=np.float64
        )

        distances = np.sort(distances)
        ts = min(int(tail_size), len(distances))

        if ts < 1:
            raise RuntimeError(
                f"OpenMax: class {c} has an empty EVT tail."
            )

        tail = np.ascontiguousarray(
            distances[-ts:],
            dtype=np.float64
        )

        mr = libmr.MR()

        # Python libMR bindings used with the original OpenMax code expose
        # fit_high(data, length).  Keep the exact tail passed to LibMR.
        mr.fit_high(
            tail.tolist(),
            len(tail)
        )

        models[c] = {
            "mav": mav,
            "mr": mr,
            "tail_size_used": int(ts),
            "n_correct_used": int(np.sum(correct_mask)),
            "n_fit_used": int(len(A_c)),
            "fallback_all_class_samples": bool(fallback_used),
        }

    return models


def openmax_single(
    activation,
    models,
    known,
    alpha_rank=10,
    euclidean_divisor=200.0
):
    """
    Compute OpenMax probabilities:
        [known class 1, ..., known class K, UNKNOWN]

    The top-alpha classifier activations are revised according to the LibMR
    w-score for their distance to the corresponding class MAV.
    """

    known = sorted(known)
    K = len(known)

    alpha_rank = min(
        max(1, int(alpha_rank)),
        K
    )

    activation = np.asarray(
        activation,
        dtype=np.float64
    )

    ranked = np.argsort(
        activation
    )[::-1]

    alpha_weights = np.zeros(
        K,
        dtype=np.float64
    )

    for rank, idx in enumerate(
        ranked[:alpha_rank]
    ):
        alpha_weights[idx] = (
            alpha_rank - rank
        ) / alpha_rank

    revised = activation.copy()
    unknown_activation = 0.0

    for local_idx, c in enumerate(known):
        model = models[c]

        dist = eucos_distance(
            activation,
            model["mav"],
            euclidean_divisor
        )

        wscore = float(
            model["mr"].w_score(
                float(dist)
            )
        )

        omega = (
            alpha_weights[local_idx]
            * wscore
        )

        removed = (
            activation[local_idx]
            * omega
        )

        revised[local_idx] = (
            activation[local_idx]
            - removed
        )

        unknown_activation += removed

    open_activations = np.r_[
        revised,
        unknown_activation
    ]

    m = np.max(
        open_activations
    )

    ex = np.exp(
        open_activations - m
    )

    probs = ex / (
        ex.sum() + 1e-12
    )

    return probs


def openmax_probabilities(
    activations,
    models,
    known,
    alpha_rank=10,
    euclidean_divisor=200.0
):
    return np.vstack(
        [
            openmax_single(
                a,
                models,
                known,
                alpha_rank,
                euclidean_divisor
            )
            for a in activations
        ]
    )



def predict_openmax(
    openmax_probs,
    known
):
    """
    Canonical-style OpenMax decision.

    The OpenMax recalibration produces K known-class probabilities plus one
    explicit UNKNOWN probability. A sample is rejected only when the explicit
    unknown probability is the largest posterior.

    No additional validation-derived rejection threshold is added.
    """

    known = np.array(
        sorted(known),
        dtype=int
    )

    known_probs = openmax_probs[
        :, :-1
    ]

    unknown_prob = openmax_probs[
        :, -1
    ]

    best_idx = known_probs.argmax(
        axis=1
    )

    best_known = known_probs[
        np.arange(len(openmax_probs)),
        best_idx
    ]

    pred = known[
        best_idx
    ].copy()

    reject = (
        unknown_prob > best_known
    )

    pred[reject] = UNKNOWN

    # AUROC convention:
    # known = positive => larger score must mean "more known-like".
    knownness_score = (
        best_known
        - unknown_prob
    )

    return pred, knownness_score


# =============================================================================
# METRICS
# =============================================================================

def metrics(y, p, score):
    kn = y != UNKNOWN
    un = ~kn

    yk = y[kn]
    pk = p[kn]
    pu = p[un]

    TP = np.sum(
        pk == yk
    )

    FN = np.sum(
        pk != yk
    )

    TU = np.sum(
        pu == UNKNOWN
    )

    FU = np.sum(
        pu != UNKNOWN
    )

    n = len(y)

    Ao = (
        (TP + TU) / n
        if n
        else 0.0
    )

    AKS = (
        TP / (TP + FN)
        if (TP + FN)
        else 0.0
    )

    AUS = (
        TU / (TU + FU)
        if (TU + FU)
        else 0.0
    )

    FAR = (
        FU / (TU + FU)
        if (TU + FU)
        else 0.0
    )

    FRR = (
        FN / (TP + FN)
        if (TP + FN)
        else 0.0
    )

    NA = 0.5 * (
        AKS + AUS
    )

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

    group_cols = [
        "Dataset", "Target_Openness", "Realized_Openness",
        "N_Known", "N_Open", "N_Total"
    ]

    for keys, g in df.groupby(group_cols):
        name, target_o, real_o, nk, no, nt = keys
        row = {
            "Dataset": name,
            "Method": "OpenMax",
            "Target_Openness": target_o,
            "Realized_Openness": real_o,
            "N_Known": nk,
            "N_Open": no,
            "N_Total": nt,
            "N_Runs": len(g)
        }

        for metric_name in METRICS:
            row[metric_name + "_mean"] = g[metric_name].mean()
            row[metric_name + "_std"] = g[metric_name].std(ddof=0)

        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["Target_Openness", "Dataset"]
    )

def print_summary(summary):
    for target_o in sorted(summary["Target_Openness"].unique()):
        q = summary[summary["Target_Openness"] == target_o]

        print(f"\nTARGET OPENNESS = {target_o:.2f}")
        print(
            f"{'Dataset':<20}{'Real.O':>8}{'K/U':>9}"
            f"{'Ao':>8}{'AKS':>8}{'AUS':>8}{'NA':>8}"
            f"{'AUROC':>8}{'F1':>8}"
        )
        print("-" * 86)

        for _, r in q.iterrows():
            ku = f"{int(r['N_Known'])}/{int(r['N_Open'])}"
            print(
                f"{r['Dataset']:<20}"
                f"{r['Realized_Openness']:>8.3f}"
                f"{ku:>9}"
                f"{r['Ao_mean']:>8.4f}"
                f"{r['AKS_mean']:>8.4f}"
                f"{r['AUS_mean']:>8.4f}"
                f"{r['NA_mean']:>8.4f}"
                f"{r['AUROC_mean']:>8.4f}"
                f"{r['F1_macro_mean']:>8.4f}"
            )


# =============================================================================
# CONFIG OBJECT (settings are overridden by scripts/run_experiment.py)
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

    args.tail_size = TAIL_SIZE
    args.alpha_rank = ALPHA_RANK
    args.euclidean_divisor = EUCOS_EUCLIDEAN_DIVISOR
    args.device = DEVICE

    return args


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = build_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available on this machine.")

    device = torch.device(
        "cuda"
        if (
            args.device == "cuda"
            or (args.device == "auto" and torch.cuda.is_available())
        )
        else "cpu"
    )

    print("Device:", device)
    print("Data directory:", args.data_dir)
    print(
        "Runs:", args.n_runs,
        "| Max epochs:", args.epochs,
        "| Batch:", args.batch_size,
        "| Tail:", args.tail_size,
        "| Alpha:", args.alpha_rank,
        "| Eucos divisor:", args.euclidean_divisor
    )
    print("Target openness levels:", TARGET_OPENNESS_LEVELS)

    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(
            f"Data directory not found:\n{args.data_dir}\n\n"
            "Check --data-dir or the DATA_DIR setting."
        )

    rows = []

    for ds in DATASETS:
        ds_name = ds["name"]

        try:
            Xdf, y = load_scrna(
                os.path.join(args.data_dir, ds["matrix"]),
                os.path.join(args.data_dir, ds["labels"]),
                ds_name
            )
        except Exception as e:
            print("SKIP", ds_name, type(e).__name__, e)
            continue

        classes = sorted(np.unique(y))
        n_total = len(classes)

        if n_total != ds["n_total"]:
            print(
                f"UYARI {ds_name}: configured total={ds['n_total']}, "
                f"actual total={n_total}. Realized openness actual total ile hesaplanacak."
            )

        for target_openness in TARGET_OPENNESS_LEVELS:
            cfg = N_KNOWN_BY_OPENNESS.get(ds_name, {})

            if target_openness not in cfg:
                print(
                    f"\n[{ds_name}] Target O={target_openness:.2f} SKIP: "
                    "meaningful multiclass configuration not available."
                )
                continue

            n_known = cfg[target_openness]
            n_open = n_total - n_known

            if n_known < 2 or n_known >= n_total:
                print(
                    f"\n[{ds_name}] Target O={target_openness:.2f} SKIP: "
                    f"n_known={n_known}, n_total={n_total}"
                )
                continue

            openness = realized_openness(n_known, n_total)

            print("\n" + "#" * 96)
            print(
                f"{ds_name} | Target O={target_openness:.2f} | "
                f"Realized O={openness:.4f} | "
                f"{n_known} known / {n_open} open | total={n_total}"
            )
            print("#" * 96)

            for run in range(args.n_runs):
                seed = args.base_seed + 17 * run
                class_seed = args.base_seed + 13 * run

                rng = np.random.default_rng(class_seed)

                known = sorted(
                    rng.choice(classes, n_known, replace=False).tolist()
                )
                open_ = [c for c in classes if c not in known]

                print(
                    f"\n[{ds_name}] O_target={target_openness:.2f} "
                    f"O_real={openness:.4f} Run {run + 1}/{args.n_runs} "
                    f"Known={known} Open={open_}"
                )

                try:
                    io, iv, itk, itu = split_indices(
                        y, known, open_, seed,
                        args.test_size, args.val_size,
                        args.unknown_sampling
                    )

                    Xo, Xv, Xk, Xu, genes = preprocess_train_only(
                        Xdf, io, iv, itk, itu, args.n_top_genes
                    )

                    yo = y[io]
                    yv = y[iv]
                    yk = y[itk]

                    global_to_local = {
                        c: i for i, c in enumerate(known)
                    }

                    yol = np.array(
                        [global_to_local[c] for c in yo],
                        dtype=int
                    )
                    yvl = np.array(
                        [global_to_local[c] for c in yv],
                        dtype=int
                    )

                    model, best_epoch = train_model(
                        Xo, yol, Xv, yvl,
                        len(known), device, seed, args
                    )

                    _, Ao, pred_o_local = get_activations(model, Xo, device)
                    _, Ak, _ = get_activations(model, Xk, device)
                    _, Au, _ = get_activations(model, Xu, device)

                    om_models = fit_openmax(
                        Ao, yo, pred_o_local, known,
                        args.tail_size,
                        args.euclidean_divisor
                    )

                    Atest = np.vstack([Ak, Au])
                    ytest = np.r_[
                        yk,
                        np.full(len(Au), UNKNOWN, dtype=int)
                    ]

                    Ptest = openmax_probabilities(
                        Atest, om_models, known,
                        args.alpha_rank,
                        args.euclidean_divisor
                    )

                    pred, score = predict_openmax(Ptest, known)
                    metric_values = metrics(ytest, pred, score)

                    tail_used = [
                        om_models[c]["tail_size_used"] for c in known
                    ]
                    correct_used = [
                        om_models[c]["n_correct_used"] for c in known
                    ]

                    row = {
                        "Dataset": ds_name,
                        "Method": "OpenMax",
                        "Run": run + 1,
                        "Seed": seed,
                        "Class_Seed": class_seed,
                        "Best_Epoch": best_epoch,
                        "Target_Openness": target_openness,
                        "Realized_Openness": openness,
                        "N_Known": n_known,
                        "N_Open": n_open,
                        "N_Total": n_total,
                        "Known_Classes": "|".join(map(str, known)),
                        "Open_Classes": "|".join(map(str, open_)),
                        "Tail_Size_Set": args.tail_size,
                        "Tail_Size_Mean_Used": float(np.mean(tail_used)),
                        "Correct_Train_Per_Class_Mean": float(
                            np.mean(correct_used)
                        ),
                        "Fallback_Classes": int(
                            sum(
                                om_models[c]["fallback_all_class_samples"]
                                for c in known
                            )
                        ),
                        "Alpha_Rank": args.alpha_rank,
                        "Eucos_Euclidean_Divisor": args.euclidean_divisor,
                        **metric_values
                    }

                    rows.append(row)

                    print(
                        f"  -> Ao={metric_values['Ao']:.4f} "
                        f"AKS={metric_values['AKS']:.4f} "
                        f"AUS={metric_values['AUS']:.4f} "
                        f"NA={metric_values['NA']:.4f} "
                        f"AUROC={metric_values['AUROC']:.4f} "
                        f"F1={metric_values['F1_macro']:.4f}"
                    )

                    # Incremental save.
                    pd.DataFrame(rows).to_csv(
                        os.path.join(
                            args.output_dir,
                            "openmax_openness_020_030_all_runs.csv"
                        ),
                        index=False
                    )

                except Exception as e:
                    print("  RUN ERROR:", type(e).__name__, e)

    if rows:
        df = pd.DataFrame(rows)
        summary = summarize(df)

        all_runs_path = os.path.join(
            args.output_dir,
            "openmax_openness_020_030_all_runs.csv"
        )
        summary_path = os.path.join(
            args.output_dir,
            "openmax_openness_020_030_summary.csv"
        )

        df.to_csv(all_runs_path, index=False)
        summary.to_csv(summary_path, index=False)

        print_summary(summary)

        print("\nOutput folder:", os.path.abspath(args.output_dir))
        print("All-runs CSV:", all_runs_path)
        print("Summary CSV:", summary_path)

    else:
        print("\nNo successful run was produced.")


if __name__ == "__main__":
    main()
