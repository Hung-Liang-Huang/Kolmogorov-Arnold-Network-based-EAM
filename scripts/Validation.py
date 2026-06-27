"""
Validation.py

Validate a trained KAN-EAM/MEAM model by comparing predicted
energies and forces with DFT reference data.

Main outputs:
    1. Relative energy parity plot by category
    2. Absolute energy parity plot
    3. Force parity plot
    4. Energy and force RMSE
    5. CSV files
    6. Text summary

    
Author:
    Huang, Hung-Liang
"""

from __future__ import annotations

from pathlib import Path
import sys
import os
import csv
import shutil
from collections import defaultdict

# ============================================================
# Prevent __pycache__ as much as possible
# This must be before importing Training_concave.
# ============================================================

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))


# ============================================================
# Import model and dataset definitions from training code
# ============================================================
#
# IMPORTANT:
# Your file must be:
#
#     Training_concave.py
#
# in the same folder as this Validation.py.
#
# Correct:
#     from Training_concave import ...
#
# Wrong:
#     from Training_concave.py import ...
#
# ============================================================

from Training_concave import (
    DEVICE,
    DTYPE,
    KANEAM_Cubic,
    FrameDataset,
    make_species_info,
    scan_species_from_npz_files,
)


# ============================================================
# User settings
# ============================================================

VALIDATION_DATA_ROOT = SCRIPT_DIR / "val"

CHECKPOINT_PATH = SCRIPT_DIR / "BEST_BALANCED.pt"

VALIDATION_OUT = SCRIPT_DIR / "validation_out_kan_eam_Ag_stage1_BEST_BALANCED"
VALIDATION_OUT.mkdir(parents=True, exist_ok=True)

PLOT_DPI = 250

MAX_FORCE_POINTS_FOR_PLOT = 200000

RANDOM_SEED = 12345

# Remove automatically generated cache folder after validation
REMOVE_PYCACHE_AFTER_RUN = True

# This folder may be created when importing Training_concave.py.
# It will be removed only if it contains no files.
REMOVE_EMPTY_TRAINING_OUT_FOLDER = True
UNWANTED_TRAINING_OUT = SCRIPT_DIR / "training_out_kan_eam_Ag_all10_fixed_rho"


# ============================================================
# Cleanup helper functions
# ============================================================

def folder_contains_files(folder: Path) -> bool:
    if not folder.exists():
        return False

    for p in folder.rglob("*"):
        if p.is_file():
            return True

    return False


def remove_folder_if_empty_or_only_empty_dirs(folder: Path):
    if not folder.exists():
        return

    if folder_contains_files(folder):
        print(f"[INFO] Not removing folder because it contains files: {folder}")
        return

    shutil.rmtree(folder, ignore_errors=True)
    print(f"[INFO] Removed empty folder: {folder}")


def cleanup_generated_import_folders():
    if REMOVE_PYCACHE_AFTER_RUN:
        pycache = SCRIPT_DIR / "__pycache__"
        if pycache.exists():
            shutil.rmtree(pycache, ignore_errors=True)
            print(f"[INFO] Removed __pycache__: {pycache}")

    if REMOVE_EMPTY_TRAINING_OUT_FOLDER:
        remove_folder_if_empty_or_only_empty_dirs(UNWANTED_TRAINING_OUT)


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

    if "05_vacancy_and_point_defects" in p:
        return "05_vacancy_and_point_defects"



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


def finite_xy(x: np.ndarray, y: np.ndarray):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    finite = np.isfinite(x) & np.isfinite(y)
    return x[finite], y[finite]


def get_plot_limits(x: np.ndarray, y: np.ndarray, pad_fraction: float = 0.06):
    vmin = min(float(np.min(x)), float(np.min(y)))
    vmax = max(float(np.max(x)), float(np.max(y)))

    pad = pad_fraction * (vmax - vmin + 1.0e-12)

    return vmin - pad, vmax + pad


def make_parity_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    out_path: Path,
    rmse_text: str | None = None,
    point_size: float = 18.0,
    alpha: float = 0.65,
):
    y_true, y_pred = finite_xy(y_true, y_pred)

    if y_true.size == 0:
        print(f"[WARN] No data for plot: {out_path}")
        return

    vmin, vmax = get_plot_limits(y_true, y_pred)

    plt.figure(figsize=(6.4, 6.2))

    plt.scatter(
        y_true,
        y_pred,
        s=point_size,
        alpha=alpha,
        edgecolors="black",
        linewidths=0.25,
    )

    plt.plot([vmin, vmax], [vmin, vmax], "k--", linewidth=1.2, label="Ideal")

    if rmse_text is not None:
        plt.text(
            0.04,
            0.94,
            rmse_text,
            transform=plt.gca().transAxes,
            fontsize=12,
            va="top",
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"),
        )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.xlim(vmin, vmax)
    plt.ylim(vmin, vmax)
    plt.gca().set_aspect("equal", adjustable="box")

    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=PLOT_DPI)
    plt.close()


def make_relative_energy_parity_by_category(
    e_true_pa: np.ndarray,
    e_pred_pa: np.ndarray,
    categories: list[str],
    out_path: Path,
):
    """
    Make a plot like your attached figure.

    Relative energy is calculated by subtracting the minimum DFT energy
    per atom from both DFT and KAN-EAM energies:

        E_DFT_relative  = E_DFT  - min(E_DFT)
        E_pred_relative = E_pred - min(E_DFT)

    Unit: meV/atom
    """

    e_true_pa = np.asarray(e_true_pa, dtype=np.float64)
    e_pred_pa = np.asarray(e_pred_pa, dtype=np.float64)
    categories = np.asarray(categories, dtype=object)

    finite = np.isfinite(e_true_pa) & np.isfinite(e_pred_pa)
    e_true_pa = e_true_pa[finite]
    e_pred_pa = e_pred_pa[finite]
    categories = categories[finite]

    if e_true_pa.size == 0:
        print(f"[WARN] No data for relative energy plot: {out_path}")
        return

    e0 = float(np.min(e_true_pa))

    e_true_rel_mev = (e_true_pa - e0) * 1000.0
    e_pred_rel_mev = (e_pred_pa - e0) * 1000.0

    energy_rmse_mev = rmse(e_pred_rel_mev - e_true_rel_mev)

    vmin, vmax = get_plot_limits(e_true_rel_mev, e_pred_rel_mev)

    plt.figure(figsize=(7.6, 6.8))

    unique_categories = sorted(set(categories.tolist()))

    for cat in unique_categories:
        mask = categories == cat

        plt.scatter(
            e_true_rel_mev[mask],
            e_pred_rel_mev[mask],
            s=45,
            alpha=0.80,
            edgecolors="black",
            linewidths=0.55,
            label=cat,
        )

    plt.plot(
        [vmin, vmax],
        [vmin, vmax],
        "k--",
        linewidth=1.6,
        label="Ideal",
    )

    plt.text(
        0.04,
        0.94,
        f"Energy RMSE = {energy_rmse_mev:.3f} meV/atom",
        transform=plt.gca().transAxes,
        fontsize=14,
        va="top",
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"),
    )

    plt.xlabel("DFT relative energy (meV/atom)")
    plt.ylabel("KAN-EAM relative energy (meV/atom)")
    plt.title("Relative energy parity by category")

    plt.xlim(vmin, vmax)
    plt.ylim(vmin, vmax)
    plt.gca().set_aspect("equal", adjustable="box")

    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=PLOT_DPI)
    plt.close()

    print(f"[DONE] Relative energy parity by category: {out_path}")


def make_force_parity_by_category(
    force_true_by_cat: dict[str, list[np.ndarray]],
    force_pred_by_cat: dict[str, list[np.ndarray]],
    out_path: Path,
):
    all_true = []
    all_pred = []
    all_cat = []

    for cat in sorted(force_true_by_cat.keys()):
        if len(force_true_by_cat[cat]) == 0:
            continue

        ft = np.concatenate(force_true_by_cat[cat]).astype(np.float64)
        fp = np.concatenate(force_pred_by_cat[cat]).astype(np.float64)

        all_true.append(ft)
        all_pred.append(fp)
        all_cat.extend([cat] * ft.size)

    if len(all_true) == 0:
        print(f"[WARN] No force data for force-by-category plot: {out_path}")
        return

    all_true = np.concatenate(all_true)
    all_pred = np.concatenate(all_pred)
    all_cat = np.asarray(all_cat, dtype=object)

    finite = np.isfinite(all_true) & np.isfinite(all_pred)
    all_true = all_true[finite]
    all_pred = all_pred[finite]
    all_cat = all_cat[finite]

    if all_true.size > MAX_FORCE_POINTS_FOR_PLOT:
        ids = np.random.choice(
            all_true.size,
            size=MAX_FORCE_POINTS_FOR_PLOT,
            replace=False,
        )

        all_true = all_true[ids]
        all_pred = all_pred[ids]
        all_cat = all_cat[ids]

    force_rmse_value = rmse(all_pred - all_true)

    vmin, vmax = get_plot_limits(all_true, all_pred)

    plt.figure(figsize=(7.2, 6.6))

    for cat in sorted(set(all_cat.tolist())):
        mask = all_cat == cat

        plt.scatter(
            all_true[mask],
            all_pred[mask],
            s=10,
            alpha=0.45,
            edgecolors="black",
            linewidths=0.20,
            label=cat,
        )

    plt.plot(
        [vmin, vmax],
        [vmin, vmax],
        "k--",
        linewidth=1.5,
        label="Ideal",
    )

    plt.text(
        0.04,
        0.94,
        f"Force RMSE = {force_rmse_value:.3e} eV/Å",
        transform=plt.gca().transAxes,
        fontsize=12,
        va="top",
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"),
    )

    plt.xlabel("DFT force component (eV/Å)")
    plt.ylabel("KAN-EAM force component (eV/Å)")
    plt.title("Force parity by category")

    plt.xlim(vmin, vmax)
    plt.ylim(vmin, vmax)
    plt.gca().set_aspect("equal", adjustable="box")

    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=PLOT_DPI)
    plt.close()

    print(f"[DONE] Force parity by category: {out_path}")


def write_energy_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as f:
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
                "true_relative_energy_meV_per_atom",
                "pred_relative_energy_meV_per_atom",
                "error_relative_energy_meV_per_atom",
                "source_path",
            ]
        )

        true_e_pa = np.asarray([row["E_true_pa"] for row in rows], dtype=np.float64)
        e0 = float(np.min(true_e_pa))

        for i, row in enumerate(rows):
            true_rel = (row["E_true_pa"] - e0) * 1000.0
            pred_rel = (row["E_pred_pa"] - e0) * 1000.0

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
                    true_rel,
                    pred_rel,
                    pred_rel - true_rel,
                    row["path"],
                ]
            )


def write_force_csv(path: Path, force_true: np.ndarray, force_pred: np.ndarray):
    force_true = np.asarray(force_true, dtype=np.float64)
    force_pred = np.asarray(force_pred, dtype=np.float64)

    with path.open("w", newline="", encoding="utf-8") as f:
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
    path.write_text("\n".join(summary_lines), encoding="utf-8")


def check_required_paths():
    if not VALIDATION_DATA_ROOT.exists():
        raise FileNotFoundError(
            "\nCannot find validation data folder:\n"
            f"    {VALIDATION_DATA_ROOT}\n\n"
            "Expected folder structure:\n\n"
            "    Desktop/\n"
            "    ├── Validation.py\n"
            "    ├── Training_concave.py\n"
            "    ├── BEST_BALANCED.pt\n"
            "    └── val/\n"
        )

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            "\nCannot find checkpoint file:\n"
            f"    {CHECKPOINT_PATH}\n\n"
            "Expected checkpoint:\n"
            "    BEST_BALANCED.pt\n"
        )


# ============================================================
# Load model
# ============================================================

def load_trained_model(checkpoint_path: Path, data_root: Path):
    print(f"[INFO] Loading checkpoint: {checkpoint_path}")

    ckpt = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )

    if "model_state" not in ckpt:
        raise KeyError(
            "The checkpoint does not contain key 'model_state'.\n"
            "Please check whether BEST_BALANCED.pt was saved by the expected training code."
        )

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

    check_required_paths()

    print("============================================================")
    print(" KAN-EAM Ag validation")
    print("============================================================")
    print(f"[INFO] Script folder:    {SCRIPT_DIR}")
    print(f"[INFO] Validation data:  {VALIDATION_DATA_ROOT}")
    print(f"[INFO] Checkpoint:       {CHECKPOINT_PATH}")
    print(f"[INFO] Output folder:    {VALIDATION_OUT}")
    print(f"[INFO] Device:           {DEVICE}")
    print("============================================================")

    model, sp, ckpt = load_trained_model(
        checkpoint_path=CHECKPOINT_PATH,
        data_root=VALIDATION_DATA_ROOT,
    )

    ds = FrameDataset(VALIDATION_DATA_ROOT, sp)

    print(f"[INFO] Number of validation structures: {len(ds)}")

    if len(ds) == 0:
        raise RuntimeError(
            "The validation dataset is empty. Please check whether the folder "
            f"{VALIDATION_DATA_ROOT} contains valid .npz validation files."
        )

    energy_rows = []

    all_e_true_pa = []
    all_e_pred_pa = []
    all_e_err_pa = []
    all_categories = []

    all_f_true = []
    all_f_pred = []
    all_f_err = []

    category_e_err = defaultdict(list)
    category_f_err = defaultdict(list)
    category_nframes = defaultdict(int)
    category_nforce_components = defaultdict(int)

    force_true_by_cat = defaultdict(list)
    force_pred_by_cat = defaultdict(list)

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

        # Do NOT use torch.no_grad(), because force prediction requires autograd.
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
        all_categories.append(category)

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

            force_true_by_cat[category].append(F_true.reshape(-1))
            force_pred_by_cat[category].append(F_pred.reshape(-1))

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
    # Overall metrics
    # ============================================================

    E_RMSE = rmse(all_e_err_pa)
    E_MAE = mae(all_e_err_pa)
    E_R2 = r2_score(all_e_true_pa, all_e_pred_pa)

    F_RMSE = rmse(all_f_err)
    F_MAE = mae(all_f_err)
    F_R2 = r2_score(all_f_true, all_f_pred)

    E_RMSE_meV = E_RMSE * 1000.0
    E_MAE_meV = E_MAE * 1000.0

    print("\n============================================================")
    print(" Overall validation metrics")
    print("============================================================")
    print(f"Energy RMSE = {E_RMSE:.8e} eV/atom")
    print(f"Energy RMSE = {E_RMSE_meV:.6f} meV/atom")
    print(f"Energy MAE  = {E_MAE:.8e} eV/atom")
    print(f"Energy MAE  = {E_MAE_meV:.6f} meV/atom")
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
            f"Nforce={nforce:10d} | "
            f"E_RMSE={e_rmse_cat * 1000.0:.6f} meV/atom | "
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

    with category_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "category",
                "n_frames",
                "n_force_components",
                "energy_rmse_eV_per_atom",
                "energy_rmse_meV_per_atom",
                "energy_mae_eV_per_atom",
                "energy_mae_meV_per_atom",
                "force_rmse_eV_per_A",
                "force_mae_eV_per_A",
            ]
        )

        for row in category_summary_rows:
            cat, nframes, nforce, e_rmse_cat, e_mae_cat, f_rmse_cat, f_mae_cat = row

            writer.writerow(
                [
                    cat,
                    nframes,
                    nforce,
                    e_rmse_cat,
                    e_rmse_cat * 1000.0,
                    e_mae_cat,
                    e_mae_cat * 1000.0,
                    f_rmse_cat,
                    f_mae_cat,
                ]
            )

    # ============================================================
    # Plots
    # ============================================================

    make_relative_energy_parity_by_category(
        e_true_pa=all_e_true_pa,
        e_pred_pa=all_e_pred_pa,
        categories=all_categories,
        out_path=VALIDATION_OUT / "parity_relative_energy_by_category.png",
    )

    make_parity_plot(
        y_true=all_e_true_pa,
        y_pred=all_e_pred_pa,
        xlabel="DFT energy per atom (eV/atom)",
        ylabel="KAN-EAM energy per atom (eV/atom)",
        title="Absolute energy parity",
        rmse_text=f"Energy RMSE = {E_RMSE_meV:.3f} meV/atom",
        out_path=VALIDATION_OUT / "parity_energy_per_atom.png",
        point_size=35.0,
        alpha=0.80,
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
            title="Force parity",
            rmse_text=f"Force RMSE = {F_RMSE:.3e} eV/Å",
            out_path=VALIDATION_OUT / "parity_force_components.png",
            point_size=10.0,
            alpha=0.45,
        )

        make_force_parity_by_category(
            force_true_by_cat=force_true_by_cat,
            force_pred_by_cat=force_pred_by_cat,
            out_path=VALIDATION_OUT / "parity_force_components_by_category.png",
        )

    plt.figure(figsize=(7.0, 5.0))
    plt.hist(all_e_err_pa * 1000.0, bins=50, alpha=0.8)
    plt.xlabel("Energy error per atom: Pred - DFT (meV/atom)")
    plt.ylabel("Count")
    plt.title(f"Energy error distribution\nRMSE = {E_RMSE_meV:.3f} meV/atom")
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
    summary_lines.append(f"Script folder: {SCRIPT_DIR}")
    summary_lines.append(f"Validation data root: {VALIDATION_DATA_ROOT}")
    summary_lines.append(f"Checkpoint: {CHECKPOINT_PATH}")
    summary_lines.append(f"Number of structures: {len(ds)}")
    summary_lines.append("")
    summary_lines.append("Overall metrics")
    summary_lines.append("------------------------------------------------------------")
    summary_lines.append(f"Energy RMSE = {E_RMSE:.10e} eV/atom")
    summary_lines.append(f"Energy RMSE = {E_RMSE_meV:.10f} meV/atom")
    summary_lines.append(f"Energy MAE  = {E_MAE:.10e} eV/atom")
    summary_lines.append(f"Energy MAE  = {E_MAE_meV:.10f} meV/atom")
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
            f"E_RMSE={e_rmse_cat * 1000.0:.10f} meV/atom "
            f"E_MAE={e_mae_cat * 1000.0:.10f} meV/atom "
            f"F_RMSE={f_rmse_cat:.10e} eV/A "
            f"F_MAE={f_mae_cat:.10e} eV/A"
        )

    summary_txt = VALIDATION_OUT / "validation_summary.txt"
    write_summary_txt(summary_txt, summary_lines)

    print("\n============================================================")
    print("[DONE] Validation finished.")
    print(f"[DONE] Summary:      {summary_txt}")
    print(f"[DONE] Energy CSV:   {energy_csv}")

    if all_f_true.size > 0:
        print(f"[DONE] Force CSV:    {force_csv}")
    else:
        print("[DONE] Force CSV:    Not written because no force data were found.")

    print(f"[DONE] Category CSV: {category_csv}")
    print(f"[DONE] Plots saved in: {VALIDATION_OUT}")
    print("============================================================")

    cleanup_generated_import_folders()


if __name__ == "__main__":
    main()