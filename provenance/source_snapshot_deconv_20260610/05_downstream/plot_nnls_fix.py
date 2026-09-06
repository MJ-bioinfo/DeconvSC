import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, numpy as np
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "lib")); import plot_style  # noqa: F401  (Arial font applied on import + plot_style.save/emf_dir for EMF)
OUT="/disk1/maijl/deconv/deconv_20260605/downstream/beyond_centroid"
methods=["Route2\nsoftmax","Route2\nNNLS+core","Δz\n(ref)"]
comp_sd=[0.0002,0.0176,0.0034]; id_bulk=[14,43,71]; id_model=[43,57,43]
cols=["#E67E22","#C0392B","#2C3E50"]
fig,ax=plt.subplots(1,2,figsize=(11,4.3))
ax[0].bar(methods,comp_sd,color=cols,edgecolor="black")
for i,v in enumerate(comp_sd): ax[0].text(i,v+0.0004,f"{v:.4f}",ha="center",fontweight="bold",fontsize=9)
ax[0].axhline(0.0002,color="gray",ls="--",lw=0.8); ax[0].set_ylabel("composition SD across 7 COVID samples")
ax[0].set_title("(A) Composition resolution\n(higher = samples have distinct proportions)",fontsize=10,fontweight="bold")
x=np.arange(3); w=0.36
ax[1].bar(x-w/2,id_bulk,w,label="norm over bulk genes",color="#5DADE2",edgecolor="black")
ax[1].bar(x+w/2,id_model,w,label="norm over model genes",color="#48C9B0",edgecolor="black")
ax[1].axhline(100/7,color="gray",ls="--",lw=1,label="chance (14%)")
ax[1].set_xticks(x); ax[1].set_xticklabels(methods); ax[1].set_ylabel("sample identity vs bulk (%)"); ax[1].set_ylim(0,80)
ax[1].set_title("(B) Sample identity vs real bulk",fontsize=10,fontweight="bold"); ax[1].legend(fontsize=8)
for i in range(3):
    ax[1].text(i-w/2,id_bulk[i]+1,id_bulk[i],ha="center",fontsize=8); ax[1].text(i+w/2,id_model[i]+1,id_model[i],ha="center",fontsize=8)
fig.suptitle("NNLS+core fixes Route2's degenerate COVID composition (GSE159585)",fontweight="bold",y=1.02)
plt.tight_layout(); plt.savefig(f"{OUT}/nnls_fix_covid.png",dpi=160,bbox_inches="tight"); print("[saved] nnls_fix_covid.png")
