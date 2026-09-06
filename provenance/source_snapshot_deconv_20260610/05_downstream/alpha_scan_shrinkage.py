#!/usr/bin/env python
"""Scan alpha in Route2 COVID generation to test the shrinkage explanation.
e_t(b_s) = psi(mu_t) + alpha * delta(b_s,t). For each alpha, regenerate COVID cells (train residual
ONCE) and measure: (1) whole-pseudobulk identity vs real bulk, (2) composition-free identity,
(3) per-cell-type profile fidelity to real COVID cells, (4) marker gene-gene co-expression preservation."""
import sys, os, json, numpy as np, pandas as pd, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+"/..")
import run_eval_HCA_fold2 as R
import route2_lib as r2
import anndata as ad, scipy.sparse as sp, scanpy as sc
DEV=r2.device("cuda:0"); torch.manual_seed(0); np.random.seed(0)
BULK="/disk1/maijl/deconv/data/GSE159585/GSE159585_bulk_RNA_seq_raw_counts_filtered.txt"
GE="/disk1/maijl/deconv/cVAE/GSE159585/covid"
REF="/disk1/maijl/deconv/data/GSE159585/GSE159585_covidset.h5ad"
OUT="/disk1/maijl/deconv/deconv_20260605/downstream/beyond_centroid"
SAMPLES=['GC1003242','GC1003243','GC1003244','GC1003245','GC1003246','GC1003247','GC1003248']
MID,CPS=256,2000
ALPHAS=[0.0,0.5,1.0,2.0,4.0,8.0]

def solve_props(basis_sig,target_sig,names,steps=300,temp=0.5,le=0.1):
    lg=torch.zeros(basis_sig.shape[0],requires_grad=True,device=DEV); opt=torch.optim.Adam([lg],lr=0.05)
    for _ in range(steps):
        opt.zero_grad(); p=F.softmax(lg/temp,0); ps=(p.view(-1,1)*basis_sig).sum(0)
        vx,vy=ps-ps.mean(),target_sig-target_sig.mean(); pcc=(vx*vy).sum()/(vx.norm()*vy.norm()+1e-8)
        (((1-pcc)+le*(p*torch.log(p+1e-8)).sum()+0.01*(lg**2).sum())).backward(); opt.step()
    p=F.softmax(lg/temp,0).detach().cpu().numpy(); p[p<1e-4]=0; return {names[i]:float(p[i]/ (p.sum()+1e-8)) for i in range(len(names))}

# ---- setup (once) ----
genes=pd.read_csv(f"{GE}/prediction_genes.csv")["prediction_genes"].astype(str).to_numpy()
G=len(genes); gidx={g:i for i,g in enumerate(genes)}
INPUT_SIZE=int(json.load(open(f"{GE}/meta.json"))["input_size"])
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
id2name,corrs,dropped=r2.recover_id2name(cent_by_id,tmean,train_types,n_types); names=[id2name[i] for i in range(n_types)]
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
sig_idx=torch.tensor([gidx[g] for g in muni if g in gidx],device=DEV); mk_idx=np.array([gidx[g] for g in muni if g in gidx])
gmask,w=r2.build_marker_core_mask(G,core,muni,gidx)
net=r2.ResidualNet(len(core),n_types,G,torch.tensor(gmask).to(DEV)).to(DEV)
Xb,Xt,td=r2.build_examples(train_donors,list(id2name.values()),{n:i for i,n in id2name.items()},dt_lin,dt_log,core,np.stack([cent_by_id[i] for i in range(n_types)]))
r2.train_residual(net,Xb,Xt,td,torch.tensor(w),DEV); net.eval()
print(f"[setup] G={G} types={n_types} core={len(core)} markers={len(mk_idx)} donors={len(train_donors)}")

# bulk + per-sample deltas (once)
bulk=pd.read_csv(BULK,sep="\t",index_col=0).T; bulk=bulk.loc[:,~bulk.columns.duplicated()].reindex(columns=genes).fillna(0.0)
bulk=bulk.div(bulk.sum(1),axis=0)*1e4
samples=[s for s in SAMPLES if s in bulk.index.astype(str).tolist()]
deltas_s={}
for s in samples:
    bcore=torch.tensor(np.log1p(bulk.loc[s].values.astype(np.float32))[core],device=DEV).unsqueeze(0)
    with torch.no_grad(): deltas_s[s]={t:net(bcore,torch.tensor([t],device=DEV)).reshape(-1) for t in range(n_types)}

# real bulk (truth) log1p-CP10K
bulkT=np.log1p(bulk.loc[samples].T)   # genes x samples (already CP10K)
def wsig(M): return M.sub(M.mean(1),axis=0)
def identity(P):  # P: genes x samples (df)
    B=wsig(bulkT.loc[P.index]).values; M=wsig(P).values; n=P.shape[1]
    C=np.array([[np.corrcoef(M[:,i],B[:,j])[0,1] for j in range(n)] for i in range(n)])
    return float(np.mean([C[i,i]==C[i].max() for i in range(n)]))

# real COVID reference: per-celltype mean + marker gene-gene corr
realX=Xtr.toarray() if sp.issparse(Xtr) else np.asarray(Xtr); realct=tr.obs["labels"].values
real_ctmean={t:realX[realct==t].mean(0) for t in train_types}
real_gg=np.corrcoef(realX[:,colmap][:, [np.where(colmap==i)[0][0] for i in mk_idx if i in colmap]].T) if False else None
# marker gene-gene corr on real (columns present)
mk_present=[i for i in mk_idx if i in set(colmap)]
col_for=lambda gi: np.where(colmap==gi)[0][0]
realM=realX[:, [col_for(i) for i in mk_present]]
real_gg=np.corrcoef(realM.T); iu=np.triu_indices_from(real_gg,1)

rows=[]
for alpha in ALPHAS:
    gen_cells=[]; gen_ct=[]; ps_whole={}; ps_cfree={}
    for s in samples:
        d=deltas_s[s]
        with torch.no_grad():
            basis=torch.stack([torch.expm1(torch.clamp(cent_mat[t]+alpha*d[t],min=0)) for t in range(n_types)])
            tgt=torch.tensor(bulk.loc[s].values,device=DEV)
        fr=solve_props(basis[:,sig_idx],tgt[sig_idx],names)
        with torch.no_grad():
            cells=[]; cmean=[]
            for t in range(n_types):
                nm=names[t]
                if nm.startswith("__unmatched"): continue
                cnt=int(round(fr.get(nm,0.0)*CPS))
                prof=torch.clamp(cent_mat[t]+alpha*d[t],min=0)  # per-celltype mean profile (log space)
                cmean.append(prof.cpu().numpy())
                if cnt<=0: continue
                mu=pri[t][0].to(DEV); std=torch.exp(0.5*pri[t][1].to(DEV))
                z=mu+torch.randn(cnt,mu.shape[1],device=DEV)*std
                lab=F.one_hot(torch.tensor([t]*cnt,device=DEV),n_types).float()
                cl=torch.clamp(vae.decode(z,lab)+alpha*d[t],min=0).cpu().numpy().astype(np.float32)
                gen_cells.append(cl); gen_ct.extend([nm]*cnt); cells.append(cl)
            allc=np.vstack(cells)
            ps_whole[s]=allc.mean(0)                       # composition-weighted
            ps_cfree[s]=np.mean(cmean,0)                   # composition-free (mean over cell-type profiles)
    Pw=pd.DataFrame(ps_whole,index=genes); Pc=pd.DataFrame(ps_cfree,index=genes)
    idw=identity(Pw) if alpha>0 else identity(Pw)         # whole has proportion variation even at a=0
    try: idc=identity(Pc)
    except: idc=np.nan
    # fidelity: per-celltype profile vs real, gene-wise corr avg over cell types
    Xg=np.vstack(gen_cells); gct=np.array(gen_ct)
    fid=[]
    for t in train_types:
        if (gct==t).sum()>=10:
            gm=Xg[gct==t].mean(0); fid.append(np.corrcoef(gm,real_ctmean[t])[0,1])
    prof_fid=float(np.nanmean(fid))
    # co-expression: marker gene-gene corr preservation (pooled)
    genM=Xg[:, [col_for(i) for i in mk_present]]
    gg=np.corrcoef(genM.T); coexp=float(np.corrcoef(gg[iu],real_gg[iu])[0,1])
    rows.append(dict(alpha=alpha, identity_whole=idw*100, identity_cfree=idc*100, profile_fidelity=prof_fid, coexpression=coexp))
    print(f"alpha={alpha}: id_whole={idw*100:.0f}% id_cfree={idc*100:.0f}% prof_fid={prof_fid:.3f} coexp={coexp:.3f}")

df=pd.DataFrame(rows); df.to_csv(f"{OUT}/alpha_scan_shrinkage.csv",index=False)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "lib")); import plot_style  # noqa: F401  (Arial font applied on import + plot_style.save/emf_dir for EMF)
fig,ax=plt.subplots(1,2,figsize=(12,4.6))
ax[0].plot(df.alpha,df.identity_whole,"-o",color="#C0392B",label="whole pseudobulk")
ax[0].plot(df.alpha,df.identity_cfree,"-s",color="#E67E22",label="composition-free")
ax[0].axhline(100/7,color="gray",ls="--",label="chance"); ax[0].axvline(1,color="k",ls=":",lw=0.8,label="default α=1")
ax[0].set_xlabel("α (residual scale)"); ax[0].set_ylabel("sample identity vs bulk (%)"); ax[0].set_title("(A) Sample identity vs α",fontweight="bold"); ax[0].legend(fontsize=8); ax[0].set_ylim(0,100)
ax[1].plot(df.alpha,df.profile_fidelity,"-o",color="#27AE60",label="per-celltype profile vs real")
ax[1].plot(df.alpha,df.coexpression,"-s",color="#2C3E50",label="marker co-expression preservation")
ax[1].axvline(1,color="k",ls=":",lw=0.8); ax[1].set_xlabel("α (residual scale)"); ax[1].set_ylabel("fidelity to real (corr)"); ax[1].set_title("(B) Biological/structure fidelity vs α",fontweight="bold"); ax[1].legend(fontsize=8)
fig.suptitle("α scan: does turning up the residual recover sample identity? (Route2 COVID, GSE159585)",fontweight="bold",y=1.02)
plt.tight_layout(); plt.savefig(f"{OUT}/alpha_scan_shrinkage.png",dpi=160,bbox_inches="tight")
print(f"[saved] {OUT}/alpha_scan_shrinkage.png")
