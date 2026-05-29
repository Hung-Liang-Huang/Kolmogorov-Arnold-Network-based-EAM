"""
validate.py

Validate a trained KAN-EAM/MEAM model by comparing predicted
energies and forces with DFT reference data.

Main outputs:
    Energy parity plot
    Force parity plot
    Energy RMSE
    Force RMSE

Author:
    Huang, Hung-Liang
"""



from __future__ import annotations

from pathlib import Path
import math
import csv
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt


# ============================================================
# Import model and dataset definitions from your training code
# ============================================================

from train_KAN_EAM_Ag_stage1 import (
    DEVICE,
    DTYPE,
    DATA_ROOT,
    OUT_ROOT,
    KANEAM_Cubic,
    SpeciesInfo,
    FrameDataset,
    make_species_info,
    scan_species_from_npz_files,
)


# ============================================================
# User settings
# ============================================================

VALIDATION_DATA_ROOT = Path("./training_data_Ag_stage1")

CHECKPOINT_PATH = Path(
    "./training_out_kan_eam_Ag_stage1_01_02_03/checkpoints/BEST_BALANCED.pt"
)

VALIDATION_OUT = Path(
    "./validation_out_kan_eam_Ag_stage1_BEST_BALANCED"
)

VALIDATION_OUT.mkdir(parents=True, exist_ok=True)

PLOT_DPI = 250

# To avoid huge CSV files, force parity data can be subsampled for plotting.
# RMSE still uses all force components.
MAX_FORCE_POINTS_FOR_PLOT = 200000

RANDOM_SEED = 12345


# ============================================================
# Helper functions
# ============================================================

def category_from_path(path: str) -> str:
    p = path.replace("\\", "/").lower()

    if "01_anchor_fcc_eos" in p:
        return "01_anchor_fcc_eos"

    if "02_anchor_fcc_small_displacement" in p:
        return "02_anchor_fcc_small_displacement"

    if "03_anchor_fcc_elastic_strain" in p:
        return "03_anchor_fcc_elastic_strain"

    if "04_fcc_finite_temperature_aimd" in p:
        return "04_fcc_finite_temperature_AIMD"

    if "05_competing_crystal_structures" in p:
        return "05_competing_crystal_structures"

    if "06_vacancy_and_point_defects" in p:
        return "06_vacancy_and_point_defects"

    if "07_surfaces" in p:
        return "07_surfaces"

    if "08_stacking_faults_and_shear" in p:
        return "08_stacking_faults_and_shear"

    if "09_short_range_repulsion" in p:
        return "09_short_range_repulsion"

    if "10_high_temperature_liquid_or_disordered" in p:
        return "10_high_temperature_liquid_or_disordered"

    return "unknown"


def rmse(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(x * x)))


def mae(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return float("nan")
    return float(np.mean(np.abs(x)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if y_true.size == 0:
        return float("nan")

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot < 1.0e-30:
        return float("nan")

    return float(1.0 - ss_res / ss_tot)


def make_parity_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    out_path: Path,
):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[finite]
    y_pred = y_pred[finite]

    if y_true.size == 0:
        print(f"[WARN] No data for plot: {out_path}")
        return

    vmin = min(float(np.min(y_true)), float(np.min(y_pred)))
    vmax = max(float(np.max(y_true)), float(np.max(y_pred)))

    pad = 0.05 * (vmax - vmin + 1.0e-12)
    vmin -= pad
    vmax += pad

    plt.figure(figsize=(6.0, 6.0))
    plt.scatter(y_true, y_pred, s=8, alpha=0.45, linewidths=0)
    plt.plot([vmin, vmax], [vmin, vmax], "k--", linewidth=1.2)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.xlim(vmin, vmax)
    plt.ylim(vmin, vmax)
    plt.gca().set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plt.savefig(out_path, dpi=PLOT_DPI)
    plt.close()


def write_energy_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "index",
                "category",
                "natoms",
                "true_energy_eV",
                "pred_energy_eV",
                "true_energy_per_atom_eV",
                "pred_energy_per_atom_eV",
                "error_energy_per_atom_eV",
                "source_path",
            ]
        )

        for i, row in enumerate(rows):
            writer.writerow(
                [
                    i,
                    row["category"],
                    row["natoms"],
                    row["E_true"],
                    row["E_pred"],
                    row["E_true_pa"],
                    row["E_pred_pa"],
                    row["E_err_pa"],
                    row["path"],
                ]
            )


def write_force_csv(path: Path, force_true: np.ndarray, force_pred: np.ndarray):
    force_true = np.asarray(force_true, dtype=np.float64)
    force_pred = np.asarray(force_pred, dtype=np.float64)

    with path.open("w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "component_index",
                "true_force_eV_per_A",
                "pred_force_eV_per_A",
                "error_force_eV_per_A",
            ]
        )

        for i in range(force_true.size):
            writer.writerow(
                [
                    i,
                    force_true[i],
                    force_pred[i],
                    force_pred[i] - force_true[i],
                ]
            )


def write_summary_txt(path: Path, summary_lines: list[str]):
    path.write_text("\n".join(summary_lines))


# ============================================================
# Load model
# ============================================================

def load_trained_model(checkpoint_path: Path, data_root: Path):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Cannot find checkpoint: {checkpoint_path}")

    print(f"[INFO] Loading checkpoint: {checkpoint_path}")

    ckpt = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )

    # Build species information from the validation data.
    # For Ag-only, this should be Z=47.
    all_z = scan_species_from_npz_files(data_root)
    sp = make_species_info(all_z)

    print(f"[INFO] Validation species: {sp.z_to_type}")

    model = KANEAM_Cubic(sp).to(DEVICE).to(DTYPE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return model, sp, ckpt


# ============================================================
# Main validation
# ============================================================

def main():
    np.random.seed(RANDOM_SEED)

    print("============================================================")
    print(" KAN-EAM Ag validation")
    print("============================================================")
    print(f"[INFO] Validation data: {VALIDATION_DATA_ROOT}")
    print(f"[INFO] Checkpoint:      {CHECKPOINT_PATH}")
    print(f"[INFO] Output folder:   {VALIDATION_OUT}")
    print(f"[INFO] Device:          {DEVICE}")
    print("============================================================")

    model, sp, ckpt = load_trained_model(
        checkpoint_path=CHECKPOINT_PATH,
        data_root=VALIDATION_DATA_ROOT,
    )

    ds = FrameDataset(VALIDATION_DATA_ROOT, sp)

    print(f"[INFO] Number of validation structures: {len(ds)}")

    energy_rows = []

    all_e_true_pa = []
    all_e_pred_pa = []
    all_e_err_pa = []

    all_f_true = []
    all_f_pred = []
    all_f_err = []

    category_e_err = defaultdict(list)
    category_f_err = defaultdict(list)
    category_nframes = defaultdict(int)
    category_nforce_components = defaultdict(int)

    for idx in range(len(ds)):
        batch = ds[idx]

        category = category_from_path(batch["path"])

        cell = torch.tensor(batch["cell"], device=DEVICE, dtype=DTYPE)
        pbc = torch.tensor(batch["pbc"], device=DEVICE)
        pos = torch.tensor(
            batch["pos"],
            device=DEVICE,
            dtype=DTYPE,
            requires_grad=True,
        )
        types = torch.tensor(batch["types"], device=DEVICE, dtype=torch.long)

        E_true = float(batch["energy"])
        F_true = batch["forces"]

        natoms = int(batch["natoms"])

        # Do NOT use torch.no_grad(), because forces require autograd.
        E_pred_t, F_pred_t = model.forward_energy_forces(
            cell,
            pbc,
            pos,
            types,
            return_rhobar=False,
        )

        E_pred = float(E_pred_t.detach().cpu().numpy())

        E_true_pa = E_true / natoms
        E_pred_pa = E_pred / natoms
        E_err_pa = E_pred_pa - E_true_pa

        all_e_true_pa.append(E_true_pa)
        all_e_pred_pa.append(E_pred_pa)
        all_e_err_pa.append(E_err_pa)

        category_e_err[category].append(E_err_pa)
        category_nframes[category] += 1

        energy_rows.append(
            {
                "category": category,
                "natoms": natoms,
                "E_true": E_true,
                "E_pred": E_pred,
                "E_true_pa": E_true_pa,
                "E_pred_pa": E_pred_pa,
                "E_err_pa": E_err_pa,
                "path": batch["path"],
            }
        )

        if F_true is not None:
            F_pred = F_pred_t.detach().cpu().numpy()
            F_true = np.asarray(F_true, dtype=np.float64)

            ferr = F_pred - F_true

            all_f_true.append(F_true.reshape(-1))
            all_f_pred.append(F_pred.reshape(-1))
            all_f_err.append(ferr.reshape(-1))

            category_f_err[category].extend(ferr.reshape(-1).tolist())
            category_nforce_components[category] += ferr.size

        if (idx + 1) % 20 == 0 or (idx + 1) == len(ds):
            print(f"[INFO] Validated {idx + 1}/{len(ds)} structures")

    all_e_true_pa = np.asarray(all_e_true_pa, dtype=np.float64)
    all_e_pred_pa = np.asarray(all_e_pred_pa, dtype=np.float64)
    all_e_err_pa = np.asarray(all_e_err_pa, dtype=np.float64)

    if len(all_f_true) > 0:
        all_f_true = np.concatenate(all_f_true).astype(np.float64)
        all_f_pred = np.concatenate(all_f_pred).astype(np.float64)
        all_f_err = np.concatenate(all_f_err).astype(np.float64)
    else:
        all_f_true = np.array([], dtype=np.float64)
        all_f_pred = np.array([], dtype=np.float64)
        all_f_err = np.array([], dtype=np.float64)

    # ============================================================
    # Metrics
    # ============================================================

    E_RMSE = rmse(all_e_err_pa)
    E_MAE = mae(all_e_err_pa)
    E_R2 = r2_score(all_e_true_pa, all_e_pred_pa)

    F_RMSE = rmse(all_f_err)
    F_MAE = mae(all_f_err)
    F_R2 = r2_score(all_f_true, all_f_pred)

    print("\n============================================================")
    print(" Overall validation metrics")
    print("============================================================")
    print(f"Energy RMSE = {E_RMSE:.8e} eV/atom")
    print(f"Energy MAE  = {E_MAE:.8e} eV/atom")
    print(f"Energy R2   = {E_R2:.8f}")
    print(f"Force RMSE  = {F_RMSE:.8e} eV/Angstrom")
    print(f"Force MAE   = {F_MAE:.8e} eV/Angstrom")
    print(f"Force R2    = {F_R2:.8f}")

    # ============================================================
    # Category metrics
    # ============================================================

    categories = sorted(category_nframes.keys())

    category_summary_rows = []

    print("\n============================================================")
    print(" Category metrics")
    print("============================================================")

    for cat in categories:
        e_err_cat = np.asarray(category_e_err[cat], dtype=np.float64)
        f_err_cat = np.asarray(category_f_err[cat], dtype=np.float64)

        e_rmse_cat = rmse(e_err_cat)
        e_mae_cat = mae(e_err_cat)
        f_rmse_cat = rmse(f_err_cat)
        f_mae_cat = mae(f_err_cat)

        nframes = category_nframes[cat]
        nforce = category_nforce_components[cat]

        category_summary_rows.append(
            [
                cat,
                nframes,
                nforce,
                e_rmse_cat,
                e_mae_cat,
                f_rmse_cat,
                f_mae_cat,
            ]
        )

        print(
            f"{cat:40s} | "
            f"Nframe={nframes:6d} | "
            f"E_RMSE={e_rmse_cat:.8e} eV/atom | "
            f"F_RMSE={f_rmse_cat:.8e} eV/A"
        )

    # ============================================================
    # Save CSV files
    # ============================================================

    energy_csv = VALIDATION_OUT / "energy_predictions.csv"
    force_csv = VALIDATION_OUT / "force_predictions_components.csv"
    category_csv = VALIDATION_OUT / "category_metrics.csv"

    write_energy_csv(energy_csv, energy_rows)

    if all_f_true.size > 0:
        write_force_csv(force_csv, all_f_true, all_f_pred)

    with category_csv.open("w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "category",
                "n_frames",
                "n_force_components",
                "energy_rmse_eV_per_atom",
                "energy_mae_eV_per_atom",
                "force_rmse_eV_per_A",
                "force_mae_eV_per_A",
            ]
        )

        writer.writerows(category_summary_rows)

    # ============================================================
    # Parity plots
    # ============================================================

    make_parity_plot(
        y_true=all_e_true_pa,
        y_pred=all_e_pred_pa,
        xlabel="DFT energy per atom (eV/atom)",
        ylabel="KAN-EAM energy per atom (eV/atom)",
        title=f"Energy parity\nRMSE = {E_RMSE:.3e} eV/atom",
        out_path=VALIDATION_OUT / "parity_energy_per_atom.png",
    )

    if all_f_true.size > 0:
        if all_f_true.size > MAX_FORCE_POINTS_FOR_PLOT:
            ids = np.random.choice(
                all_f_true.size,
                size=MAX_FORCE_POINTS_FOR_PLOT,
                replace=False,
            )

            f_true_plot = all_f_true[ids]
            f_pred_plot = all_f_pred[ids]

        else:
            f_true_plot = all_f_true
            f_pred_plot = all_f_pred

        make_parity_plot(
            y_true=f_true_plot,
            y_pred=f_pred_plot,
            xlabel="DFT force component (eV/Å)",
            ylabel="KAN-EAM force component (eV/Å)",
            title=f"Force parity\nRMSE = {F_RMSE:.3e} eV/Å",
            out_path=VALIDATION_OUT / "parity_force_components.png",
        )

    # ============================================================
    # Error histogram plots
    # ============================================================

    plt.figure(figsize=(7.0, 5.0))
    plt.hist(all_e_err_pa, bins=50, alpha=0.8)
    plt.xlabel("Energy error per atom: Pred - DFT (eV/atom)")
    plt.ylabel("Count")
    plt.title(f"Energy error distribution\nRMSE = {E_RMSE:.3e} eV/atom")
    plt.tight_layout()
    plt.savefig(VALIDATION_OUT / "hist_energy_error_per_atom.png", dpi=PLOT_DPI)
    plt.close()

    if all_f_err.size > 0:
        plt.figure(figsize=(7.0, 5.0))
        plt.hist(all_f_err, bins=100, alpha=0.8)
        plt.xlabel("Force component error: Pred - DFT (eV/Å)")
        plt.ylabel("Count")
        plt.title(f"Force error distribution\nRMSE = {F_RMSE:.3e} eV/Å")
        plt.tight_layout()
        plt.savefig(VALIDATION_OUT / "hist_force_component_error.png", dpi=PLOT_DPI)
        plt.close()

    # ============================================================
    # Text summary
    # ============================================================

    summary_lines = []
    summary_lines.append("KAN-EAM Ag validation summary")
    summary_lines.append("============================================================")
    summary_lines.append(f"Validation data root: {VALIDATION_DATA_ROOT}")
    summary_lines.append(f"Checkpoint: {CHECKPOINT_PATH}")
    summary_lines.append(f"Number of structures: {len(ds)}")
    summary_lines.append("")
    summary_lines.append("Overall metrics")
    summary_lines.append("------------------------------------------------------------")
    summary_lines.append(f"Energy RMSE = {E_RMSE:.10e} eV/atom")
    summary_lines.append(f"Energy MAE  = {E_MAE:.10e} eV/atom")
    summary_lines.append(f"Energy R2   = {E_R2:.10f}")
    summary_lines.append(f"Force RMSE  = {F_RMSE:.10e} eV/Angstrom")
    summary_lines.append(f"Force MAE   = {F_MAE:.10e} eV/Angstrom")
    summary_lines.append(f"Force R2    = {F_R2:.10f}")
    summary_lines.append("")
    summary_lines.append("Category metrics")
    summary_lines.append("------------------------------------------------------------")

    for row in category_summary_rows:
        cat, nframes, nforce, e_rmse_cat, e_mae_cat, f_rmse_cat, f_mae_cat = row

        summary_lines.append(
            f"{cat:40s} "
            f"Nframe={nframes:6d} "
            f"Nforce={nforce:10d} "
            f"E_RMSE={e_rmse_cat:.10e} eV/atom "
            f"F_RMSE={f_rmse_cat:.10e} eV/A"
        )

    summary_txt = VALIDATION_OUT / "validation_summary.txt"
    write_summary_txt(summary_txt, summary_lines)

    print("\n============================================================")
    print("[DONE] Validation finished.")
    print(f"[DONE] Summary:      {summary_txt}")
    print(f"[DONE] Energy CSV:   {energy_csv}")
    print(f"[DONE] Force CSV:    {force_csv}")
    print(f"[DONE] Category CSV: {category_csv}")
    print(f"[DONE] Plots saved in: {VALIDATION_OUT}")
    print("============================================================")


if __name__ == "__main__":
    main()