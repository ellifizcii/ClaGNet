from pathlib import Path
# -*- coding: utf-8 -*-
"""
CLaGNet: Contrastive Latent Gaussian Network for open-set scRNA-seq annotation
Leakage-free Residual MLP + Supervised Contrastive Loss + class-wise GMM.

Requirements:
    pip install numpy pandas scikit-learn torch

Outputs:
    deep_openset_gmm_all_runs.csv
    deep_openset_gmm_summary.csv
"""

import os, json, random, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.metrics import f1_score, roc_auc_score, roc_curve





warnings.filterwarnings("ignore")
UNKNOWN = -1
METRICS = ["Ao", "AKS", "AUS", "NA", "FAR", "FRR", "F1_macro", "AUROC"]



DATA_DIR = str(Path(__file__).resolve().parents[1] / "data")
OUTPUT_DIR = "./deep_openset_results"

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
LAMBDA_SUPCON = 0.20
TEMPERATURE = 0.10
MAX_GMM_COMPONENTS = 4
FPR_CONSTRAINT = 0.05

DEVICE = "auto"   # "auto", "cpu", "cuda"

DATASETS = [
    {"name": "Li",            "matrix": "data_li.csv", "labels": "labs_li.csv",            "n_total": 9,  "n_known": 6},
    {"name": "Zeisel",        "matrix": "data_zeisel.csv", "labels": "labs_zeisel.csv",    "n_total": 9,  "n_known": 6},
    {"name": "Tasic",         "matrix": "data_tasic.csv", "labels": "labs_tasic.csv",       "n_total": 18, "n_known": 12},
    {"name": "Darmanis",      "matrix": "data_darmanis.csv", "labels": "labs_darmanis.csv", "n_total": 9,  "n_known": 6},
    {"name": "Xin",           "matrix": "data_xin.csv", "labels": "labs_xin.csv",           "n_total": 8,  "n_known": 5},
    {"name": "Kolodziejczyk", "matrix": "data_kolodziejczyk.csv", "labels": "labs_kolodziejczyk.csv", "n_total": 3, "n_known": 2},
    {"name": "Baron Mouse",   "matrix": "data_baronm1.csv", "labels": "labs_baronm1.csv",   "n_total": 13, "n_known": 9},
    {"name": "Pollen",        "matrix": "data_pollen.csv", "labels": "labs_pollen.csv",      "n_total": 11, "n_known": 7},
]


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def load_scrna(matrix_path, labels_path, name):
    mat = pd.read_csv(matrix_path)
    genes = mat.iloc[:, 0].astype(str)
    expr = mat.iloc[:, 1:].copy(); expr.index = genes; expr = expr.T
    expr.index = pd.Index(range(1, len(expr)+1), name="cell_id")

    lab = pd.read_csv(labels_path)
    label_col = "cell_type" if "cell_type" in lab.columns else "x"
    if "Unnamed: 0" in lab.columns:
        
        lab = lab.rename(columns={"Unnamed: 0":"cell_id"})
    elif "cell_id" not in lab.columns:
        lab["cell_id"] = np.arange(1, len(lab)+1)
    lab["cell_id"] = lab["cell_id"].astype(int); lab = lab.set_index("cell_id")

    ids = expr.index.intersection(lab.index)
    expr = expr.loc[ids]
    y_raw = lab.loc[ids, label_col].values
    uniq = sorted(np.unique(y_raw), key=lambda x: str(x))
    m = {v:i+1 for i,v in enumerate(uniq)}
    y = np.array([m[v] for v in y_raw], dtype=int)

    # Cell-wise only: safe to compute once.
    lib = expr.sum(axis=1).replace(0, 1.0)
    xlog = np.log1p(expr.div(lib, axis=0)*1e4).astype(np.float32)
    print(f"[{name}] {xlog.shape[0]} cells | {xlog.shape[1]} genes | {len(uniq)} classes")
    return xlog, y


def split_indices(y, known, open_, seed, test_size=.20, val_size=.15, unknown_sampling="balanced"):
    ki = np.where(np.isin(y, known))[0]
    ui = np.where(np.isin(y, open_))[0]
    tr, tk = train_test_split(ki, test_size=test_size, random_state=seed, stratify=y[ki])
    opt, val = train_test_split(tr, test_size=val_size, random_state=seed, stratify=y[tr])

    if unknown_sampling == "all":
        tu = ui
    else:
        rng = np.random.default_rng(seed)
        uc = np.unique(y[ui])
        per = max(1, len(tk)//2//len(uc))
        arr=[]
        for c in uc:
            ii = ui[y[ui]==c]
            arr.append(rng.choice(ii, size=min(per, len(ii)), replace=False))
        tu = np.concatenate(arr)
    return opt, val, tk, tu


def preprocess_train_only(df, opt, val, tk, tu, n_hvg=2000):
    dopt = df.iloc[opt]
    genes = dopt.var(axis=0).nlargest(min(n_hvg, dopt.shape[1])).index
    def arr(idx): return df.iloc[idx].loc[:, genes].to_numpy(np.float32, copy=False)
    a,b,c,d = arr(opt),arr(val),arr(tk),arr(tu)
    sc = StandardScaler(); a=sc.fit_transform(a).astype(np.float32)
    b=sc.transform(b).astype(np.float32); c=sc.transform(c).astype(np.float32); d=sc.transform(d).astype(np.float32)
    return a,b,c,d,genes


class DS(Dataset):
    def __init__(self,X,y): self.X=torch.tensor(X,dtype=torch.float32); self.y=torch.tensor(y,dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self,i): return self.X[i],self.y[i]


class ResBlock(nn.Module):
    def __init__(self,d,drop):
        super().__init__(); self.f1=nn.Linear(d,d); self.b1=nn.BatchNorm1d(d); self.f2=nn.Linear(d,d); self.b2=nn.BatchNorm1d(d); self.drop=nn.Dropout(drop)
    def forward(self,x):
        r=x; x=self.drop(F.gelu(self.b1(self.f1(x)))); x=self.b2(self.f2(x)); return F.gelu(x+r)


class Net(nn.Module):
    def __init__(self,input_dim,n_classes,hidden=256,latent=64,drop=.25):
        super().__init__()
        self.stem=nn.Sequential(nn.Linear(input_dim,512),nn.BatchNorm1d(512),nn.GELU(),nn.Dropout(drop),nn.Linear(512,hidden),nn.BatchNorm1d(hidden),nn.GELU())
        self.r1=ResBlock(hidden,drop); self.r2=ResBlock(hidden,drop)
        self.head=nn.Sequential(nn.Linear(hidden,128),nn.BatchNorm1d(128),nn.GELU(),nn.Dropout(drop),nn.Linear(128,latent))
        self.cls=nn.Linear(latent,n_classes)
    def forward(self,x):
        h=self.r2(self.r1(self.stem(x))); z=F.normalize(self.head(h),p=2,dim=1); return z,self.cls(z)


def supcon(z,y,temp=.10):
    if len(z)<2: return z.new_tensor(0.)
    sim=z@z.T/temp; sim=sim-sim.max(1,keepdim=True).values.detach()
    same=(y[:,None]==y[None,:]).float(); eye=torch.eye(len(z),device=z.device)
    mask=1-eye; same*=mask
    logp=sim-torch.log((torch.exp(sim)*mask).sum(1,keepdim=True)+1e-12)
    npos=same.sum(1); valid=npos>0
    if valid.sum()==0: return z.new_tensor(0.)
    return -((same*logp).sum(1)/(npos+1e-12))[valid].mean()


def train_model(Xo,yo,Xv,yv,n_classes,device,seed,args):
    seed_all(seed); model=Net(Xo.shape[1],n_classes,args.hidden_dim,args.latent_dim,args.dropout).to(device)
    dl=DataLoader(DS(Xo,yo),batch_size=min(args.batch_size,len(Xo)),shuffle=True,num_workers=0,drop_last=(len(Xo)>args.batch_size))
    vl=DataLoader(DS(Xv,yv),batch_size=min(512,len(Xv)),shuffle=False,num_workers=0)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    ce=nn.CrossEntropyLoss(); best=float('inf'); state=None; bad=0
    for ep in range(1,args.epochs+1):
        model.train()
        for xb,yb in dl:
            xb,yb=xb.to(device),yb.to(device); opt.zero_grad(set_to_none=True)
            z,logit=model(xb); loss=ce(logit,yb)+args.lambda_supcon*supcon(z,yb,args.temperature); loss.backward(); opt.step()
        model.eval(); s=n=0
        with torch.no_grad():
            for xb,yb in vl:
                xb,yb=xb.to(device),yb.to(device); _,lg=model(xb); l=ce(lg,yb); s+=l.item()*len(xb); n+=len(xb)
        v=s/max(n,1)
        if v<best-1e-5:
            best=v; state={k:t.detach().cpu().clone() for k,t in model.state_dict().items()}; bad=0
        else: bad+=1
        if ep==1 or ep%10==0: print(f"      epoch {ep:03d} val_CE={v:.4f}")
        if bad>=args.patience: break
    if state: model.load_state_dict(state)
    return model


@torch.no_grad()
def encode(model,X,device):
    model.eval(); out=[]
    dl=DataLoader(DS(X,np.zeros(len(X),int)),batch_size=min(512,len(X)),shuffle=False,num_workers=0)
    for xb,_ in dl:
        z,_=model(xb.to(device)); out.append(z.cpu().numpy())
    return np.vstack(out)


def fit_gmms(Z,y,known,max_k,seed):
    gm={}
    for c in known:
        z=Z[y==c]; bestk=1; best=np.inf
        for k in range(1,min(max_k,max(1,len(z)//5),len(z))+1):
            try:
                g=GaussianMixture(k,covariance_type='diag',reg_covar=1e-4,n_init=2,max_iter=300,random_state=seed).fit(z)
                b=g.bic(z)
                if b<best: best=b; bestk=k
            except Exception: pass
        gm[c]=GaussianMixture(bestk,covariance_type='diag',reg_covar=1e-4,n_init=3,max_iter=300,random_state=seed).fit(z)
    return gm


def score_matrix(Z,gm):
    classes=sorted(gm); S=np.column_stack([gm[c].score_samples(Z) for c in classes]); return classes,S


def calibrate(Zv,yv,gm,fpr_limit=.05):
    classes,S=score_matrix(Zv,gm); th={}
    for j,c in enumerate(classes):
        pos=S[yv==c,j]; neg=S[yv!=c,j]
        if len(pos)==0 or len(neg)==0: th[c]=np.percentile(pos,5) if len(pos) else -np.inf; continue
        yy=np.r_[np.ones(len(pos)),np.zeros(len(neg))]; ss=np.r_[pos,neg]
        fpr,tpr,t=roc_curve(yy,ss); valid=np.where(fpr<=fpr_limit)[0]
        idx=valid[np.argmax(tpr[valid])] if len(valid) else np.argmax(tpr-fpr); th[c]=float(t[idx])
    return th


def predict(Z,gm,th):
    classes,S=score_matrix(Z,gm); j=S.argmax(1); pred=np.array([classes[k] for k in j],int); best=S[np.arange(len(Z)),j]
    for i,c in enumerate(pred.copy()):
        if best[i]<th.get(c,-np.inf): pred[i]=UNKNOWN
    return pred,best


def metrics(y,p,score):
    kn=y!=UNKNOWN; un=~kn; yk,pk=y[kn],p[kn]; pu=p[un]
    TP=np.sum(pk==yk); FN=np.sum(pk!=yk); TU=np.sum(pu==UNKNOWN); FU=np.sum(pu!=UNKNOWN); n=len(y)
    Ao=(TP+TU)/n; AKS=TP/(TP+FN) if TP+FN else 0; AUS=TU/(TU+FU) if TU+FU else 0
    FAR=FU/(TU+FU) if TU+FU else 0; FRR=FN/(TP+FN) if TP+FN else 0; NA=.5*(AKS+AUS)
    labs=sorted(np.unique(y[y!=UNKNOWN])); F1=f1_score(y,p,labels=labs,average='macro',zero_division=0)
    try: A=roc_auc_score((y!=UNKNOWN).astype(int),score)
    except Exception: A=np.nan
    return dict(Ao=Ao,AKS=AKS,AUS=AUS,NA=NA,FAR=FAR,FRR=FRR,F1_macro=F1,AUROC=A)


def summarize(df):
    rows=[]
    for name,g in df.groupby('Dataset'):
        r={'Dataset':name,'Method':'CLaGNet','N_Runs':len(g)}
        for m in METRICS:
            r[m+'_mean']=g[m].mean(); r[m+'_std']=g[m].std(ddof=0)
        rows.append(r)
    return pd.DataFrame(rows)


def print_summary(s):
    print('\n                 FINAL SUMMARY — CLaGNet')
    print('                        (LEAKAGE-FREE)\n')
    print(f"{'Dataset':<20}{'Ao':>8}{'AKS':>8}{'AUS':>8}{'NA':>8}{'AUROC':>8}{'F1':>8}")
    print('-'*76)
    for _,r in s.iterrows():
        print(f"{r.Dataset:<20}{r.Ao_mean:>8.4f}{r.AKS_mean:>8.4f}{r.AUS_mean:>8.4f}{r.NA_mean:>8.4f}{r.AUROC_mean:>8.4f}{r.F1_macro_mean:>8.4f}")



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
    args.base_seed = BASE_SEED
    args.n_top_genes = N_TOP_GENES
    args.test_size = TEST_SIZE
    args.val_size = VAL_SIZE
    args.unknown_sampling = UNKNOWN_SAMPLING
    args.hidden_dim = HIDDEN_DIM
    args.latent_dim = LATENT_DIM
    args.dropout = DROPOUT
    args.epochs = EPOCHS
    args.batch_size = BATCH_SIZE
    args.lr = LR
    args.weight_decay = WEIGHT_DECAY
    args.lambda_supcon = LAMBDA_SUPCON
    args.temperature = TEMPERATURE
    args.patience = PATIENCE
    args.max_gmm_components = MAX_GMM_COMPONENTS
    args.fpr_constraint = FPR_CONSTRAINT
    args.device = DEVICE
    return args


def main():
    args = build_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available on this machine.")

    device = torch.device(
        "cuda"
        if (args.device == "cuda" or
            (args.device == "auto" and torch.cuda.is_available()))
        else "cpu"
    )

    print("Device:", device)
    print("Data directory:", args.data_dir)
    print("Runs:", args.n_runs, "| Epochs:", args.epochs, "| Patience:", args.patience)

    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(
            f"Data directory not found:\n{args.data_dir}\n\n"
            "Check --data-dir or the DATA_DIR setting."
        )

    datasets = DATASETS
    rows = []

    for ds in datasets:
        print("\n" + "#" * 90)
        print(
            f"{ds['name']} | matrix={ds['matrix']} | labels={ds['labels']} | "
            f"known={ds['n_known']} | total={ds['n_total']}"
        )
        print("#" * 90)

        try:
            Xdf, y = load_scrna(
                os.path.join(args.data_dir, ds["matrix"]),
                os.path.join(args.data_dir, ds["labels"]),
                ds["name"]
            )
        except Exception as e:
            print("SKIP", ds["name"], type(e).__name__, e)
            continue

        classes = sorted(np.unique(y))

        if len(classes) < ds["n_known"]:
            print(
                f"SKIP {ds['name']}: actual number of classes {len(classes)}, "
                f"n_known={ds['n_known']} is not possible."
            )
            continue

        for run in range(args.n_runs):
            seed = args.base_seed + 17 * run
            rng = np.random.default_rng(args.base_seed + 13 * run)

            known = sorted(
                rng.choice(classes, ds["n_known"], replace=False).tolist()
            )
            open_ = [c for c in classes if c not in known]

            print(
                f"\n[{ds['name']}] Run {run+1}/{args.n_runs} "
                f"Known={known} Open={open_}"
            )

            try:
                io, iv, itk, itu = split_indices(
                    y, known, open_, seed,
                    args.test_size, args.val_size,
                    args.unknown_sampling
                )

                Xo, Xv, Xk, Xu, _ = preprocess_train_only(
                    Xdf, io, iv, itk, itu, args.n_top_genes
                )

                yo, yv, yk = y[io], y[iv], y[itk]

                mp = {c: i for i, c in enumerate(known)}
                yol = np.array([mp[c] for c in yo])
                yvl = np.array([mp[c] for c in yv])

                model = train_model(
                    Xo, yol, Xv, yvl,
                    len(known), device, seed, args
                )

                Zo, Zv, Zk, Zu = [
                    encode(model, a, device)
                    for a in (Xo, Xv, Xk, Xu)
                ]

                gm = fit_gmms(
                    Zo, yo, known,
                    args.max_gmm_components, seed
                )

                th = calibrate(
                    Zv, yv, gm, args.fpr_constraint
                )

                Z = np.vstack([Zk, Zu])
                yt = np.r_[yk, np.full(len(Zu), UNKNOWN)]

                pred, score = predict(Z, gm, th)
                m = metrics(yt, pred, score)

                row = {
                    "Dataset": ds["name"],
                    "Method": "CLaGNet",
                    "Run": run + 1,
                    "Known_Classes": "|".join(map(str, known)),
                    "Open_Classes": "|".join(map(str, open_)),
                    **m
                }
                rows.append(row)

                print(
                    f"  -> Ao={m['Ao']:.4f} "
                    f"AKS={m['AKS']:.4f} "
                    f"AUS={m['AUS']:.4f} "
                    f"NA={m['NA']:.4f} "
                    f"AUROC={m['AUROC']:.4f} "
                    f"F1={m['F1_macro']:.4f}"
                )

                pd.DataFrame(rows).to_csv(
                    os.path.join(
                        args.output_dir,
                        "deep_openset_gmm_all_runs.csv"
                    ),
                    index=False
                )

            except Exception as e:
                print("  RUN ERROR:", type(e).__name__, e)

    if rows:
        df = pd.DataFrame(rows)
        s = summarize(df)

        df.to_csv(
            os.path.join(args.output_dir, "deep_openset_gmm_all_runs.csv"),
            index=False
        )
        s.to_csv(
            os.path.join(args.output_dir, "deep_openset_gmm_summary.csv"),
            index=False
        )

        print_summary(s)
        print("\nResults directory:", os.path.abspath(args.output_dir))
    else:
        print("\nNo runs completed successfully.")


if __name__ == "__main__":
    main()
