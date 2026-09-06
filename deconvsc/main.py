"""DeconvSC all-in-one entry (train / generate / all).

Generation = the prior-anchored method (frozen per-cell-type prior centroid + bulk-conditioned
residual) with prophead-predicted cell-type proportions. Samples are taken from --samples, or ALL
bulk columns if not given (dataset-agnostic; no hard-coded sample IDs). Imports the DeconvSC
package directly.

For the config-driven pipeline use scripts/*.py + configs/*.yaml instead; this is the single-file
programmatic equivalent.
"""
import os
import argparse
import pickle
import torch
import scanpy as sc

from deconvsc.preprocess import load_sc_data
from deconvsc.model import AttentionVAE
from deconvsc.trainer import train_vae
from deconvsc.infer import (
    train_anchored_residual,            # learn the expression residual + the prophead head
    generate_prior_anchored,            # generate cells (prior-anchored expression + prophead props)
)
from deconvsc.evaluate import evaluate_generated_data
from deconvsc.io import load_decode_params
from deconvsc.utils import set_seed, get_sample_indices_from_bulk


def parse_args():
    p = argparse.ArgumentParser(description="DeconvSC - attention cVAE pipeline (prior-anchored generation + prophead)")
    p.add_argument('--data_dir', type=str, required=True, help='Directory holding the sc/bulk files')
    p.add_argument('--output_dir', type=str, default='./output/', help='Output directory')
    p.add_argument('--sc_train', type=str, required=True, help='Reference scRNA h5ad filename (has donor + cell type)')
    p.add_argument('--bulk_test', type=str, required=True, help='Bulk matrix filename')
    p.add_argument('--mode', type=str, choices=['train', 'generate', 'all'], default='all')
    # generation
    p.add_argument('--alpha', type=float, default=1.0, help='residual strength (0 = pure-prior centroid only)')
    p.add_argument('--donor_col', type=str, default='sample', help='Donor/sample column in sc_train (residual + prophead)')
    p.add_argument('--celltype_label', type=str, default='cell type', help='Cell-type column in sc_train')
    p.add_argument('--samples', nargs='*', default=None, help='Bulk sample IDs to deconvolve (default: ALL bulk columns)')
    p.add_argument('--cells_per_celltype', type=int, default=500)
    # model / training
    p.add_argument('--mid_hidden_size', type=int, default=256)
    p.add_argument('--seed', type=int, default=18)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=512)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--device', type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device) if args.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}  |  generation: prior-anchored + prophead")
    os.makedirs(args.output_dir, exist_ok=True)

    sc_path = os.path.join(args.data_dir, args.sc_train)
    bulk_path = os.path.join(args.data_dir, args.bulk_test)
    meta_path = os.path.join(args.output_dir, 'sc_metadata.pkl')
    hidden_size_list = [4096, 2048, 1024]

    # ---------------- Phase 1: preprocess + train ----------------
    if args.mode in ['train', 'all']:
        print("\n=== Phase 1: Data Prep & Training ===")
        (dataset, X_tensor, labels, n_cell_types, mapping_dict, all_genes, real_bulk_log,
         single_cell_matrix, cell_number_target_num, signature_array, color_map,
         bulk_sample_names, final_signatures, sig_indices, core_genes, core_indices_tensor) = load_sc_data(
            sc_path, bulk_path, celltype_label=args.celltype_label, output_dir=args.output_dir)
        breed_2_list = list(mapping_dict.keys())
        with open(meta_path, 'wb') as f:
            pickle.dump({'input_dim': len(all_genes), 'X_tensor': X_tensor, 'labels': labels,
                         'n_cell_types': n_cell_types, 'real_bulk_log': real_bulk_log, 'mapping_dict': mapping_dict,
                         'common_genes': all_genes, 'cell_number_target_num': cell_number_target_num,
                         'breed_2_list': breed_2_list, 'color_map': color_map, 'bulk_sample_names': bulk_sample_names,
                         'sig_indices': sig_indices, 'core_genes': core_genes, 'core_indices_tensor': core_indices_tensor}, f)
        scvae = AttentionVAE(input_size=len(core_genes), output_size=len(all_genes),
                             hidden_size_list=hidden_size_list, mid_hidden_size=args.mid_hidden_size,
                             num_cell_types=n_cell_types, embedding_dim=32, nhead=4, num_layers=4).to(device)
        scvae, _ = train_vae(vae_model=scvae, X_tensor=X_tensor, labels=labels, real_bulk_log=real_bulk_log,
                             batch_size=args.batch_size, epoch_num=args.epochs, learning_rate=args.lr,
                             hidden_list=hidden_size_list, mid_hidden_size=args.mid_hidden_size,
                             num_cell_types=n_cell_types, breed_2_list=breed_2_list, color_map=color_map,
                             used_device=device, core_indices_tensor=core_indices_tensor,
                             output_dir=args.output_dir, seed=args.seed)

    # ---------------- Phase 2: deconvolution / generation ----------------
    if args.mode in ['generate', 'all']:
        print("\n=== Phase 2: Deconvolution & Generation (prior-anchored + prophead) ===")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata not found at {meta_path}; run with --mode train first.")
        with open(meta_path, 'rb') as f:
            md = pickle.load(f)
        X_tensor, labels = md['X_tensor'], md['labels']
        n_cell_types, real_bulk_log = md['n_cell_types'], md['real_bulk_log']
        mapping_dict, common_genes = md['mapping_dict'], md['common_genes']
        sig_indices, core_genes = md['sig_indices'], md['core_genes']

        scvae = AttentionVAE(input_size=len(core_genes), output_size=len(common_genes),
                             hidden_size_list=hidden_size_list, mid_hidden_size=args.mid_hidden_size,
                             num_cell_types=n_cell_types, embedding_dim=32, nhead=4, num_layers=4).to(device)
        scvae = load_decode_params(scvae, os.path.join(args.output_dir, 'scvae_best.pth'), device=device)
        priors = torch.load(os.path.join(args.output_dir, 'cell_type_mu_logvar_best.pt'), map_location=device)

        # samples: explicit list, else ALL bulk columns (dataset-agnostic, no hard-coded IDs)
        target_samples = args.samples if args.samples else list(md['bulk_sample_names'])
        sample_indices, valid_sample_names = get_sample_indices_from_bulk(bulk_path, target_samples)
        if not sample_indices:
            raise ValueError("No valid bulk samples found.")
        input_bulk_log = real_bulk_log[sample_indices]
        total_cells = n_cell_types * args.cells_per_celltype

        core_idx = [list(common_genes).index(g) for g in core_genes if g in set(common_genes)]
        residual_net, centroids_by_id, core_idx, prop_head, prop_names = train_anchored_residual(
            vae=scvae, priors=priors, sc_ref_path=sc_path, donor_col=args.donor_col,
            celltype_label=args.celltype_label, mapping_dict=mapping_dict, common_genes=common_genes,
            core_indices=core_idx, device=device, n_types=n_cell_types, seed=args.seed)
        adata_generated, _, obs_df = generate_prior_anchored(
            vae=scvae, real_bulk_tensor=input_bulk_log, sample_names=valid_sample_names,
            mapping_dict=mapping_dict, priors=priors, device=device, output_dir=args.output_dir,
            common_genes=common_genes, residual_net=residual_net, centroids_by_id=centroids_by_id,
            core_indices=core_idx, prop_head=prop_head, prop_names=prop_names, alpha=args.alpha,
            total_cells=total_cells)

        adata_generated.write_h5ad(os.path.join(args.output_dir, 'generated_data.h5ad'))
        print(f"Generation completed -> {os.path.join(args.output_dir, 'generated_data.h5ad')}")
        # Evaluate against the single-cell reference (deconv_20260610 methodology: profile/gene PCC,
        # generated-data UMAP, comparison marker heatmap). real_is_raw inferred from the value range.
        adata_real = sc.read_h5ad(sc_path)
        real_is_raw = bool(adata_real.X.max() > 50)
        evaluate_generated_data(adata_gen=adata_generated, adata_real=adata_real,
                                output_dir=args.output_dir,
                                gen_sample_col='Sample', gen_cell_col='Cell_type',
                                real_sample_col=args.donor_col, real_cell_col=args.celltype_label,
                                real_is_raw=real_is_raw, sample_level=True, seed=args.seed)
        print("Pipeline finished.")


if __name__ == "__main__":
    main()
