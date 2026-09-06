#!/usr/bin/env python
"""Compare proportion solvers for Route2 COVID (proportion-only, no cell generation).
Goal: a solver that is (a) NOT over-uniform (softmax default), (b) NOT over-sparse (NNLS),
(c) varies across samples & matches bulk identity. Reports #active cell types + identity."""
import sys, os, json, numpy as np, pandas as pd, torch, torch.nn.functional as F
from scipy.optimize import nnls
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+"/..")
import run_eval_HCA_fold2 as R; import route2_lib as r2
import anndata as ad, scipy.sparse as sp, scanpy as sc
DEV=r2.device("cuda:0"); torch.manual_seed(0); np.random.seed(0)
BULK="/disk1/maijl/deconv/data/GSE159585/GSE159585_bulk_RNA_seq_raw_counts_filtered.txt"
GE="/disk1/maijl/deconv/cVAE/GSE159585/covid"; REF="/disk1/maijl/deconv/data/GSE159585/GSE159585_covidset.h5ad"
SAMPLES=['GC1003242','GC1003243','GC1003244','GC1003245','GC1003246','GC1003247','GC1003248']; MID=256
genes=pd.read_csv(f"{GE}/prediction_genes.csv")["prediction_genes"].astype(str).to_numpy(); G=len(genes); gidx={g:i for i,g in enumerate(genes)}
INPUT_SIZE=int(json.load(open(f"{GE}/meta.json"))["input_size"]); pri=torch.load(f"{GE}/cell_type_mu_logvar_best.pt",map_location="cpu"); n_types=len(pri)
tr=ad.read_h5ad(REF); tr.obs["labels"]=tr.obs["cell type"].astype(str)
tr=tr[:,[g for g in genes if g in set(tr.var_names)]].copy(); present=tr.var_names.astype(str).to_numpy()
sc.pp.normalize_total(tr,target_sum=1e4); sc.pp.log1p(tr); colmap=np.array([gidx[g] for g in present],dtype=int)
Xtr=tr.X.tocsr() if sp.issparse(tr.X) else sp.csr_matrix(tr.X)
vae=r2.load_vae_decode(f"{GE}/scvae_best.pth",G,n_types,input_size=INPUT_SIZE,mid_hidden=MID,device=DEV)
cent_by_id=r2.decode_centroids(vae,pri,n_types,DEV); cent_mat=torch.tensor(np.stack([cent_by_id[i] for i in range(n_types)]),device=DEV)
train_types=sorted(tr.obs["labels"].unique()); tmean={}
for t in train_types:
    full=np.zeros(G,np.float32); full[colmap]=np.asarray(Xtr[tr.obs["labels"].values==t].mean(0)).ravel(); tmean[t]=full
id2name,corrs,dropped=r2.recover_id2name(cent_by_id,tmean,train_types,n_types); names=[id2name[i] for i in range(n_types)]; name2id={n:i for i,n in id2name.items()}
sc.pp.highly_variable_genes(tr,n_top_genes=3000); core=np.array([gidx[g] for g in genes if g in set(tr.var_names[tr.var["highly_variable"].values].astype(str))],dtype=int)
donors=tr.obs["Sample ID"].astype(str).values; cells_t=tr.obs["labels"].astype(str).values
dt_log,dt_lin={},{}
for d in np.unique(donors):
    dm=donors==d
    for t in np.unique(cells_t[dm]):
        mk=dm&(cells_t==t)
        if mk.sum()<5: continue
        full=np.zeros(G,np.float32); full[colmap]=np.asarray(Xtr[mk].mean(0)).ravel(); dt_log[(d,t)]=full; dt_lin[(d,t)]=np.expm1(full).astype(np.float32)
train_donors=sorted({d for (d,_) in dt_log})
markers=R.derive_marker_genes(tr,"labels",top_n=30); muni=sorted({g for v in markers.values() for g in v}); sig=np.array([gidx[g] for g in muni if g in gidx])
gmask,w=r2.build_marker_core_mask(G,core,muni,gidx); net=r2.ResidualNet(len(core),n_types,G,torch.tensor(gmask).to(DEV)).to(DEV)
Xb,Xt,td=r2.build_examples(train_donors,list(name2id),name2id,dt_lin,dt_log,core,np.stack([cent_by_id[i] for i in range(n_types)])); r2.train_residual(net,Xb,Xt,td,torch.tensor(w),DEV); net.eval()
bulk=pd.read_csv(BULK,sep="\t",index_col=0).T; bulk=bulk.loc[:,~bulk.columns.duplicated()].reindex(columns=genes).fillna(0.0); bulk=bulk.div(bulk.sum(1),axis=0)*1e4
samples=[s for s in SAMPLES if s in bulk.index.astype(str).tolist()]

def softmax_solve(B_sig,t_sig,le,temp=0.5,steps=300):
    lg=torch.zeros(B_sig.shape[0],requires_grad=True,device=DEV); opt=torch.optim.Adam([lg],lr=0.05)
    for _ in range(steps):
        opt.zero_grad(); p=F.softmax(lg/temp,0); ps=(p.view(-1,1)*B_sig).sum(0); vx,vy=ps-ps.mean(),t_sig-t_sig.mean()
        pcc=(vx*vy).sum()/(vx.norm()*vy.norm()+1e-8); (((1-pcc)+le*(p*torch.log(p+1e-8)).sum()+0.01*(lg**2).sum())).backward(); opt.step()
    p=F.softmax(lg/temp,0).detach().cpu().numpy(); p[p<1e-4]=0; return p/(p.sum()+1e-8)
def nnls_s(A,y,l2=0.0,sum1=None):
    if sum1 is not None: A=np.vstack([A, np.full((1,A.shape[1]),sum1)]); y=np.concatenate([y,[sum1]])
    if l2>0: A=np.vstack([A, l2*np.eye(A.shape[1])]); y=np.concatenate([y,np.zeros(A.shape[1])])
    p,_=nnls(A.astype(np.float64),y.astype(np.float64)); s=p.sum(); return p/s if s>0 else p

raw=pd.read_csv(BULK,sep="\t",index_col=0); raw=raw.loc[:,~raw.columns.duplicated()]
b1=np.log1p(raw.div(raw.sum(0),axis=1)*1e4)
def score(propmat):  # propmat: samples x n_types
    PS={}
    for i,s in enumerate(samples):
        with torch.no_grad():
            bl=bulk.loc[s].values.astype(np.float32); bcore=torch.tensor(np.log1p(bl)[core],device=DEV).unsqueeze(0)
            d={t:net(bcore,torch.tensor([t],device=DEV)).reshape(-1) for t in range(n_types)}
            basis=torch.stack([torch.expm1(torch.clamp(cent_mat[t]+d[t],min=0)) for t in range(n_types)])
            pb=(torch.tensor(propmat[i],device=DEV).view(-1,1)*basis).sum(0); PS[s]=np.log1p(pb.cpu().numpy())
    P=pd.DataFrame(PS,index=genes); g=b1.index.intersection(P.index)
    Bw=b1.loc[g,samples].sub(b1.loc[g,samples].mean(1),axis=0).values; Mw=P.loc[g,samples].sub(P.loc[g,samples].mean(1),axis=0).values; n=len(samples)
    C=np.array([[np.corrcoef(Mw[:,i],Bw[:,j])[0,1] for j in range(n)] for i in range(n)])
    return int(round(np.mean([C[i,i]==C[i].max() for i in range(n)])*100))

# precompute per-sample basis pieces
solvers={"softmax le=0.1":("sm",0.1),"softmax le=0.01":("sm",0.01),"softmax le=0.0":("sm",0.0),
         "nnls core":("nn",dict()),"nnls core +sum1":("nn",dict(sum1=None)),"nnls core +ridge":("nn",dict(l2=None))}
print(f"{'solver':<20} {'avg#ct>2%':>10} {'SD':>8} {'identity':>9}")
for nm,(kind,cfg) in solvers.items():
    props=[]
    for s in samples:
        with torch.no_grad():
            bl=bulk.loc[s].values.astype(np.float32); bcore=torch.tensor(np.log1p(bl)[core],device=DEV).unsqueeze(0)
            d={t:net(bcore,torch.tensor([t],device=DEV)).reshape(-1) for t in range(n_types)}
            basis=torch.stack([torch.expm1(torch.clamp(cent_mat[t]+d[t],min=0)) for t in range(n_types)])
            tgt=torch.tensor(bl,device=DEV)
        if kind=="sm":
            p=softmax_solve(basis[:,sig],tgt[sig],cfg)
        else:
            A=basis[:,core].cpu().numpy().T; y=tgt[core].cpu().numpy(); ymax=float(y.max())
            if "sum1" in cfg: p=nnls_s(A,y,sum1=ymax)
            elif "l2" in cfg: p=nnls_s(A,y,l2=0.1*ymax)
            else: p=nnls_s(A,y)
        props.append(p)
    pm=np.array(props); fr=pm/pm.sum(1,keepdims=True)
    print(f"{nm:<20} {(fr>0.02).sum(1).mean():>10.1f} {fr.std(0).mean():>8.4f} {score(pm):>8}%")
