#!/usr/bin/env python
"""OLD(Delta_z, application/) vs NEW(Route2): can the 7 COVID samples be told apart, and does each
model reproduce the REAL between-sample differences (truth = GSE159585 bulk RNA-seq, which drives
both models)? Two views: (1) whole pseudobulk (composition+expression); (2) composition-free
(equal-weight cell-type average) -> pure expression, to not penalize Route2 for weak proportions.
Metric = identity match: does generated sample i best-correlate with its OWN bulk i?"""
import anndata as ad, numpy as np, pandas as pd, scipy.sparse as sp, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "lib")); import plot_style  # noqa: F401  (Arial font applied on import + plot_style.save/emf_dir for EMF)
OUT="/disk1/maijl/deconv/deconv_20260605/downstream/beyond_centroid"
BULK="/disk1/maijl/deconv/data/GSE159585/GSE159585_bulk_RNA_seq_raw_counts_filtered.txt"
NEWp="/disk1/maijl/deconv/deconv_20260605/downstream/covid/generated_data_covid.h5ad"
OLDp="/disk1/maijl/deconv/cVAE/GSE159585/application/generated_data_covid.h5ad"
MIN=10

def loads(path):
    a=ad.read_h5ad(path); X=a.X.toarray() if sp.issparse(a.X) else np.asarray(a.X)
    g=a.var_names.astype(str).to_numpy(); s=a.obs["Sample"].astype(str).to_numpy(); c=a.obs["Cell_type"].astype(str).to_numpy()
    whole={u:X[s==u].mean(0) for u in np.unique(s)}                       # composition-weighted
    cfree={}                                                              # composition-free: mean over cell types
    for u in np.unique(s):
        cms=[X[(s==u)&(c==ct)].mean(0) for ct in np.unique(c[s==u]) if ((s==u)&(c==ct)).sum()>=MIN]
        cfree[u]=np.mean(cms,0)
    return pd.DataFrame(whole,index=g), pd.DataFrame(cfree,index=g)

bulk=pd.read_csv(BULK, sep="\t", index_col=0); bulk=bulk.loc[:, ~bulk.columns.duplicated()]
bulk=np.log1p(bulk.div(bulk.sum(0),axis=1)*1e4)
NWw,NWc=loads(NEWp); OLw,OLc=loads(OLDp)
covid=[s for s in NWw.columns if s in OLw.columns and s in bulk.columns]
g=bulk.index.intersection(NWw.index).intersection(OLw.index)
print("COVID samples:", covid)

def wsig(M): return M.sub(M.mean(1),axis=0)
def crossmatch(M):
    B=wsig(bulk.loc[g,covid]).values; Mm=wsig(M.loc[g,covid]).values
    C=np.zeros((len(covid),len(covid)))
    for i in range(len(covid)):
        for j in range(len(covid)): C[i,j]=np.corrcoef(Mm[:,i],B[:,j])[0,1]
    return C
def idmatch(C): return np.mean([C[i,i]==C[i].max() for i in range(C.shape[0])])
def diagrank(C):  # avg percentile rank of the true (diagonal) match within each row
    return np.mean([(C[i] < C[i,i]).mean() for i in range(C.shape[0])])

res={}
for tag,(Mw,Mc) in [("NEW",(NWw,NWc)),("OLD",(OLw,OLc))]:
    Cw,Cc=crossmatch(Mw),crossmatch(Mc)
    res[tag]=dict(Cw=Cw,Cc=Cc,id_w=idmatch(Cw),id_c=idmatch(Cc),dr_w=diagrank(Cw),dr_c=diagrank(Cc))
    print(f"{tag}: whole-pseudobulk identity {res[tag]['id_w']*100:.0f}% (diag-rank {res[tag]['dr_w']:.2f}); "
          f"composition-free identity {res[tag]['id_c']*100:.0f}% (diag-rank {res[tag]['dr_c']:.2f})")

# figure
lab=[s[-3:] for s in covid]
fig=plt.figure(figsize=(13,7)); gs=fig.add_gridspec(2,3, hspace=0.45, wspace=0.4)
panels=[("NEW","Cw","NEW (Route2) whole-pseudobulk"),("OLD","Cw","OLD (Delta_z) whole-pseudobulk"),
        ("NEW","Cc","NEW composition-free (expression only)"),("OLD","Cc","OLD composition-free (expression only)")]
pos=[(0,0),(0,1),(1,0),(1,1)]
for (tag,key,title),(r,cc) in zip(panels,pos):
    ax=fig.add_subplot(gs[r,cc]); C=res[tag][key]; im=ax.imshow(C,cmap="viridis")
    ax.set_xticks(range(len(lab))); ax.set_xticklabels(lab,fontsize=6,rotation=90)
    ax.set_yticks(range(len(lab))); ax.set_yticklabels(lab,fontsize=6)
    ax.set_xlabel("REAL bulk sample",fontsize=8); ax.set_ylabel(f"{tag} generated",fontsize=8)
    for i in range(len(lab)):
        j=int(C[i].argmax()); ax.add_patch(plt.Rectangle((j-0.5,i-0.5),1,1,fill=False,edgecolor="red",lw=1.5))
    idv = res[tag]['id_w'] if key=="Cw" else res[tag]['id_c']
    ax.set_title(title+"\nbest-match own bulk: "+str(int(round(idv*100)))+"%", fontsize=8.5, fontweight="bold")
    plt.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
ax=fig.add_subplot(gs[:,2])
x=np.arange(2); w=0.35
ax.bar(x-w/2,[res["NEW"]["id_w"]*100,res["NEW"]["id_c"]*100],w,label="NEW (Route2)",color="#C0392B",edgecolor="black")
ax.bar(x+w/2,[res["OLD"]["id_w"]*100,res["OLD"]["id_c"]*100],w,label="OLD (Delta_z)",color="#2C3E50",edgecolor="black")
ax.axhline(100/len(covid),color="gray",ls="--",lw=1,label=f"chance ({100/len(covid):.0f}%)")
ax.set_xticks(x); ax.set_xticklabels(["whole\npseudobulk","composition-free\n(expression)"],fontsize=9)
ax.set_ylabel("% of COVID samples whose generated profile\nbest-matches its OWN real bulk"); ax.set_ylim(0,100)
ax.set_title("Within-condition sample identity\nvs real bulk", fontsize=10, fontweight="bold"); ax.legend(fontsize=8)
fig.suptitle("Can the 7 COVID samples be told apart & matched to their REAL bulk?  Delta_z(application) vs Route2",
             fontsize=12, fontweight="bold", y=1.0)
plt.savefig(f"{OUT}/within_condition_old_vs_new.png", dpi=150, bbox_inches="tight")
print(f"[saved] {OUT}/within_condition_old_vs_new.png")
