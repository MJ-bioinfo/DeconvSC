#!/usr/bin/env python
"""Regenerate Route2 COVID with proportions from NNLS on 3000 core (HVG) genes instead of
softmax(1-PCC) on signature genes. Test: does composition stop being degenerate & do samples
become identity-matchable to their bulks? Writes covid/generated_data_covid_nnls.h5ad (no overwrite)."""
import sys, os, json, numpy as np, pandas as pd, torch, torch.nn.functional as F
from scipy.optimize import nnls
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+"/..")
import run_eval_HCA_fold2 as R
import route2_lib as r2
import anndata as ad, scipy.sparse as sp, scanpy as sc
DEV=r2.device("cuda:0"); torch.manual_seed(0); np.random.seed(0)
BULK="/disk1/maijl/deconv/data/GSE159585/GSE159585_bulk_RNA_seq_raw_counts_filtered.txt"
GE="/disk1/maijl/deconv/cVAE/GSE159585/covid"; REF="/disk1/maijl/deconv/data/GSE159585/GSE159585_covidset.h5ad"
OUTH="/disk1/maijl/deconv/deconv_20260605/downstream/covid/generated_data_covid_nnls.h5ad"
SAMPLES=['GC1003242','GC1003243','GC1003244','GC1003245','GC1003246','GC1003247','GC1003248']
MID,CPS=256,2000

# ---- setup (same as route2_generate_downstream covid) ----
genes=pd.read_csv(f"{GE}/prediction_genes.csv")["prediction_genes"].astype(str).to_numpy()
G=len(genes); gidx={g:i for i,g in enumerate(genes)}; INPUT_SIZE=int(json.load(open(f"{GE}/meta.json"))["input_size"])
pri=torch.load(f"{GE}/cell_type_mu_logvar_best.pt",map_location="cpu"); n_types=len(pri)
tr=ad.read_h5ad(REF); tr.obs["labels"]=tr.obs["cell type"].astype(str)
tr=tr[:,[g for g in genes if g in set(tr.var_names)]].copy(); present=tr.var_names.astype(str).to_numpy()
sc.pp.normalize_total(tr,target_sum=1e4); sc.pp.log1p(tr)
colmap=np.array([gidx[g] for g in present],dtype=int); Xtr=tr.X.tocsr() if sp.issparse(tr.X) else sp.csr_matrix(tr.X)
vae=r2.load_vae_decode(f"{GE}/scvae_best.pth",G,n_types,input_size=INPUT_SIZE,mid_hidden=MID,device=DEV)
cent_by_id=r2.decode_centroids(vae,pri,n_types,DEV); cent_mat=torch.tensor(np.stack([cent_by_id[i] for i in range(n_types)]),device=DEV)
train_types=sorted(tr.obs["labels"].unique()); tmean={}
for t in train_types:
    full=np.zeros(G,np.float32); full[colmap]=np.asarray(Xtr[tr.obs["labels"].values==t].mean(0)).ravel(); tmean[t]=full
id2name,corrs,dropped=r2.recover_id2name(cent_by_id,tmean,train_types,n_types); names=[id2name[i] for i in range(n_types)]; name2id={n:i for i,n in id2name.items()}
sc.pp.highly_variable_genes(tr,n_top_genes=3000); hvg=set(tr.var_names[tr.var["highly_variable"].values].astype(str))
core=np.array([gidx[g] for g in genes if g in hvg],dtype=int)
donors=tr.obs["Sample ID"].astype(str).values; cells_t=tr.obs["labels"].astype(str).values
dt_log,dt_lin={},{}
for d in np.unique(donors):
    dm=donors==d
    for t in np.unique(cells_t[dm]):
        mk=dm&(cells_t==t)
        if mk.sum()<5: continue
        full=np.zeros(G,np.float32); full[colmap]=np.asarray(Xtr[mk].mean(0)).ravel(); dt_log[(d,t)]=full; dt_lin[(d,t)]=np.expm1(full).astype(np.float32)
train_donors=sorted({d for (d,_) in dt_log})
markers=R.derive_marker_genes(tr,"labels",top_n=30); muni=sorted({g for v in markers.values() for g in v})
gmask,w=r2.build_marker_core_mask(G,core,muni,gidx)
net=r2.ResidualNet(len(core),n_types,G,torch.tensor(gmask).to(DEV)).to(DEV)
Xb,Xt,td=r2.build_examples(train_donors,list(name2id),name2id,dt_lin,dt_log,core,np.stack([cent_by_id[i] for i in range(n_types)]))
r2.train_residual(net,Xb,Xt,td,torch.tensor(w),DEV); net.eval()
print(f"[setup] G={G} types={n_types} core={len(core)} donors={len(train_donors)}")

bulk=pd.read_csv(BULK,sep="\t",index_col=0).T; bulk=bulk.loc[:,~bulk.columns.duplicated()].reindex(columns=genes).fillna(0.0); bulk=bulk.div(bulk.sum(1),axis=0)*1e4
samples=[s for s in SAMPLES if s in bulk.index.astype(str).tolist()]

def nnls_core(basis_lin, target_lin):
    A=basis_lin[:,core].detach().cpu().numpy().T.astype(np.float64)   # n_core x n_types
    y=target_lin[core].detach().cpu().numpy().astype(np.float64)
    p,_=nnls(A,y); s=p.sum(); p=p/s if s>0 else p
    return {names[t]:float(p[t]) for t in range(n_types)}

generated,meta=[],[]; props={}
for s in samples:
    bl=bulk.loc[s].values.astype(np.float32); bcore=torch.tensor(np.log1p(bl)[core],device=DEV).unsqueeze(0)
    with torch.no_grad():
        deltas={t:net(bcore,torch.tensor([t],device=DEV)).reshape(-1) for t in range(n_types)}
        basis_lin=torch.stack([torch.expm1(torch.clamp(cent_mat[t]+deltas[t],min=0)) for t in range(n_types)])
        target_lin=torch.tensor(bl,device=DEV)
        fr=nnls_core(basis_lin,target_lin); props[s]=fr
        for t in range(n_types):
            nm=names[t]
            if nm.startswith("__unmatched"): continue
            cnt=int(round(fr.get(nm,0.0)*CPS))
            if cnt<=0: continue
            mu=pri[t][0].to(DEV); std=torch.exp(0.5*pri[t][1].to(DEV))
            z=mu+torch.randn(cnt,mu.shape[1],device=DEV)*std; lab=F.one_hot(torch.tensor([t]*cnt,device=DEV),n_types).float()
            cl=torch.clamp(vae.decode(z,lab)+deltas[t],min=0).cpu().numpy().astype(np.float32)
            generated.append(cl); meta.extend([(s,nm)]*cnt)
Xg=np.vstack(generated); sm=[m[0] for m in meta]; ctv=[m[1] for m in meta]
obs=pd.DataFrame({"Sample":sm,"Cell_type":ctv,"Patient ID":sm,"cell type":ctv,"disease_status":"COVID19"}); obs.index=[f"Cell_{i}" for i in range(len(obs))]
ad.AnnData(X=Xg,obs=obs,var=pd.DataFrame(index=genes)).write_h5ad(OUTH,compression="gzip")
print(f"[saved] {len(obs)} cells -> {OUTH}")

# ---- evaluate composition SD + identity vs bulk ----
prop_df=pd.DataFrame(props).T  # samples x celltypes
print(f"\n[NNLS+core] composition SD across samples (mean over cell types) = {prop_df.std(0).mean():.4f}")
P=pd.DataFrame({s:Xg[np.array(sm)==s].mean(0) for s in samples}, index=genes)
raw=pd.read_csv(BULK,sep="\t",index_col=0); raw=raw.loc[:,~raw.columns.duplicated()]
def ident(B,M,samp):
    Bw=B.sub(B.mean(1),axis=0).values; Mw=M.sub(M.mean(1),axis=0).values; n=len(samp)
    C=np.array([[np.corrcoef(Mw[:,i],Bw[:,j])[0,1] for j in range(n)] for i in range(n)])
    return int(round(np.mean([C[i,i]==C[i].max() for i in range(n)])*100))
b1=np.log1p(raw.div(raw.sum(0),axis=1)*1e4); g=b1.index.intersection(P.index)
b2=raw.T.reindex(columns=P.index).fillna(0.0); b2=np.log1p((b2.div(b2.sum(1),axis=0)*1e4).T)
print(f"[NNLS+core] identity normOVERbulk={ident(b1.loc[g,samples],P.loc[g,samples],samples)}%  normOVERmodel={ident(b2.loc[g,samples],P.loc[g,samples],samples)}%")
print("[ref] softmax was: comp-SD=0.0002, identity 14%/43%")
