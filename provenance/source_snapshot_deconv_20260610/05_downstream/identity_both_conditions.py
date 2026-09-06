#!/usr/bin/env python
"""Sample-identity-vs-bulk test for BOTH COVID and Normal groups, OLD(Delta_z) vs NEW(Route2).
identity = generated sample i best-correlates with its OWN real bulk i (chance=1/n)."""
import anndata as ad, numpy as np, pandas as pd, scipy.sparse as sp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "lib")); import plot_style  # noqa: F401  (Arial font applied on import + plot_style.save/emf_dir for EMF)
OUT="/disk1/maijl/deconv/deconv_20260605/downstream/beyond_centroid"
BULK="/disk1/maijl/deconv/data/GSE159585/GSE159585_bulk_RNA_seq_raw_counts_filtered.txt"
P={"COVID":{"NEW":"/disk1/maijl/deconv/deconv_20260605/downstream/covid/generated_data_covid.h5ad",
            "OLD":"/disk1/maijl/deconv/cVAE/GSE159585/application/generated_data_covid.h5ad"},
   "Normal":{"NEW":"/disk1/maijl/deconv/deconv_20260605/downstream/normal_18_24/generated_data_normal_18_24.h5ad",
             "OLD":"/disk1/maijl/deconv/cVAE/GSE159585/application/generated_data_normal.h5ad"}}
MIN=10
bulk=pd.read_csv(BULK, sep="\t", index_col=0); bulk=bulk.loc[:, ~bulk.columns.duplicated()]
bulk=np.log1p(bulk.div(bulk.sum(0),axis=1)*1e4)

def loads(path):
    a=ad.read_h5ad(path); X=a.X.toarray() if sp.issparse(a.X) else np.asarray(a.X)
    g=a.var_names.astype(str).to_numpy(); s=a.obs["Sample"].astype(str).to_numpy(); c=a.obs["Cell_type"].astype(str).to_numpy()
    whole={u:X[s==u].mean(0) for u in np.unique(s)}
    cfree={u:np.mean([X[(s==u)&(c==ct)].mean(0) for ct in np.unique(c[s==u]) if ((s==u)&(c==ct)).sum()>=MIN],0) for u in np.unique(s)}
    return pd.DataFrame(whole,index=g), pd.DataFrame(cfree,index=g)

def wsig(M): return M.sub(M.mean(1),axis=0)
def identity(M, samples, gg):
    B=wsig(bulk.loc[gg,samples]).values; Mm=wsig(M.loc[gg,samples]).values
    n=len(samples); C=np.array([[np.corrcoef(Mm[:,i],B[:,j])[0,1] for j in range(n)] for i in range(n)])
    return np.mean([C[i,i]==C[i].max() for i in range(n)]), C

rows=[]
for cond in ["COVID","Normal"]:
    NWw,NWc=loads(P[cond]["NEW"]); OLw,OLc=loads(P[cond]["OLD"])
    for tag,(Mw,Mc) in [("NEW",(NWw,NWc)),("OLD",(OLw,OLc))]:
        samp=[s for s in Mw.columns if s in bulk.columns]
        gg=bulk.index.intersection(Mw.index)
        idw,_=identity(Mw,samp,gg); idc,_=identity(Mc,samp,gg)
        rows.append(dict(condition=cond, model=tag, n=len(samp), chance=100/len(samp),
                         identity_whole=idw*100, identity_expr=idc*100))
        print(f"{cond:6s} {tag}: n={len(samp)} chance={100/len(samp):.0f}% | whole {idw*100:.0f}% | expr-only {idc*100:.0f}%")
df=pd.DataFrame(rows); df.to_csv(f"{OUT}/identity_both_conditions.csv", index=False)

# figure: grouped bars
fig,axs=plt.subplots(1,2, figsize=(11,4.4), sharey=True)
for ax,metric,tt in [(axs[0],"identity_whole","whole pseudobulk (composition+expression)"),
                     (axs[1],"identity_expr","composition-free (expression only)")]:
    x=np.arange(2); w=0.35
    nv=[df[(df.condition==c)&(df.model=="NEW")][metric].values[0] for c in ["COVID","Normal"]]
    ov=[df[(df.condition==c)&(df.model=="OLD")][metric].values[0] for c in ["COVID","Normal"]]
    ax.bar(x-w/2,nv,w,label="NEW (Route2)",color="#C0392B",edgecolor="black")
    ax.bar(x+w/2,ov,w,label="OLD (Delta_z)",color="#2C3E50",edgecolor="black")
    ax.axhline(100/7,color="gray",ls="--",lw=1,label="chance (~14%)")
    ax.set_xticks(x); ax.set_xticklabels(["COVID","Normal"]); ax.set_ylim(0,100)
    ax.set_title(tt, fontsize=10, fontweight="bold"); ax.set_ylabel("% samples matched to own bulk")
    for i,(a,b) in enumerate(zip(nv,ov)):
        ax.text(i-w/2,a+1,f"{a:.0f}",ha="center",fontsize=8); ax.text(i+w/2,b+1,f"{b:.0f}",ha="center",fontsize=8)
    ax.legend(fontsize=8)
fig.suptitle("Sample identity vs real bulk — both conditions: Delta_z resolves samples, Route2 ≈ chance (consistent)", fontsize=12, fontweight="bold")
plt.tight_layout(); plt.savefig(f"{OUT}/identity_both_conditions.png", dpi=160, bbox_inches="tight")
print(f"[saved] {OUT}/identity_both_conditions.png")
