"""
export_lammps.py

Export a trained KAN-EAM/MEAM model checkpoint into a LAMMPS-compatible
EAM/fs potential file.

Expected input:
    Trained PyTorch .pt checkpoint

Expected output:
    LAMMPS-compatible .eam.fs potential file

Author:
    Huang, Hung-Liang
"""


from pathlib import Path
import numpy as np


# ============================================================
# User settings
# ============================================================

# Input exported function file from training code
# Choose BEST_BALANCED if available.
INPUT_NPZ = Path(
    "./training_out_kan_eam_Ag_stage1_01_02_03/"
    "exported_functions/"
    "KAN_EAM_FINAL.npz"
)

# Output LAMMPS eam/fs file
OUTPUT_EAMFS = Path("Ag_KAN_EAM_stage1.eam.fs")

# Element information
ELEMENT = "Ag"
Z = 47
MASS = 107.8682

# Use your relaxed DFT fcc Ag lattice constant here.
# This is metadata for LAMMPS, but should still be reasonable.
LATTICE_CONSTANT = 4.09
LATTICE_TYPE = "fcc"

# Number of grid points in final LAMMPS file.
# It can be different from training export grid.
NRHO = 10000
NR = 10000

# If True, shift F so that F(0)=0.
# Usually safe and recommended.
GAUGE_SHIFT_F_ZERO = True

# If True, force rho(rcut)=0 and phi(rcut)=0 exactly.
FORCE_TAIL_ZERO = True

# If True, clip negative density values to zero.
# Since rho(r) should be nonnegative in EAM.
CLIP_RHO_NONNEGATIVE = True

# If True, replace NaN/Inf values with finite values.
CLEAN_NUMERICAL_VALUES = True

# How many numbers per line in the EAM file
VALUES_PER_LINE = 5


# ============================================================
# Helper functions
# ============================================================

def clean_array(y, name="array"):
    y = np.asarray(y, dtype=np.float64)

    if CLEAN_NUMERICAL_VALUES:
        if not np.all(np.isfinite(y)):
            print(f"[WARN] {name} contains NaN/Inf. Replacing with finite values.")
            y = np.nan_to_num(
                y,
                nan=0.0,
                posinf=np.max(y[np.isfinite(y)]) if np.any(np.isfinite(y)) else 0.0,
                neginf=np.min(y[np.isfinite(y)]) if np.any(np.isfinite(y)) else 0.0,
            )

    return y


def interp_to_grid(x_old, y_old, x_new, name="array"):
    x_old = np.asarray(x_old, dtype=np.float64)
    y_old = np.asarray(y_old, dtype=np.float64)
    x_new = np.asarray(x_new, dtype=np.float64)

    order = np.argsort(x_old)
    x_old = x_old[order]
    y_old = y_old[order]

    y_new = np.interp(x_new, x_old, y_old)

    return clean_array(y_new, name=name)


def write_values(f, values, values_per_line=5):
    values = np.asarray(values, dtype=np.float64).ravel()

    for i in range(0, len(values), values_per_line):
        chunk = values[i:i + values_per_line]
        line = "".join(f"{v:24.16E}" for v in chunk)
        f.write(line + "\n")


def print_array_info(name, arr):
    arr = np.asarray(arr, dtype=np.float64)
    print(
        f"{name:20s}: "
        f"min={np.min(arr): .8e}, "
        f"max={np.max(arr): .8e}, "
        f"first={arr[0]: .8e}, "
        f"last={arr[-1]: .8e}"
    )


# ============================================================
# Main conversion
# ============================================================

def main():
    if not INPUT_NPZ.exists():
        raise FileNotFoundError(f"Cannot find input file: {INPUT_NPZ}")

    print("============================================================")
    print(" Convert KAN-EAM exported npz to LAMMPS eam/fs")
    print("============================================================")
    print(f"Input npz:  {INPUT_NPZ}")
    print(f"Output file:{OUTPUT_EAMFS}")
    print("============================================================")

    d = np.load(INPUT_NPZ, allow_pickle=True)

    required_keys = [
        "r_grid",
        "rho_grid",
        "type_to_z",
        "F_type0_y",
        "rho_0_to_0_y",
        "phi_0_0_y",
    ]

    for key in required_keys:
        if key not in d.files:
            raise KeyError(
                f"Missing key '{key}' in {INPUT_NPZ}. "
                f"Available keys are: {d.files}"
            )

    type_to_z = d["type_to_z"].astype(int)

    if len(type_to_z) != 1:
        raise ValueError(
            "This script is for single-element Ag only. "
            f"Detected type_to_z = {type_to_z}"
        )

    if int(type_to_z[0]) != Z:
        print(
            f"[WARN] Expected Z={Z} for Ag, but npz has Z={int(type_to_z[0])}."
        )

    # ------------------------------------------------------------
    # Read exported KAN-EAM functions
    # ------------------------------------------------------------
    r_old = clean_array(d["r_grid"], "r_grid")
    rho_old = clean_array(d["rho_grid"], "rho_grid")

    F_old = clean_array(d["F_type0_y"], "F_type0_y")
    rho_func_old = clean_array(d["rho_0_to_0_y"], "rho_0_to_0_y")
    phi_old = clean_array(d["phi_0_0_y"], "phi_0_0_y")

    rmin_export = float(np.min(r_old))
    rmax_export = float(np.max(r_old))
    rhomin_export = float(np.min(rho_old))
    rhomax_export = float(np.max(rho_old))

    if rmin_export > 1.0e-8:
        print(
            f"[INFO] Training r_grid starts at r = {rmin_export:.6f} Å. "
            "LAMMPS table will start at r = 0. "
            "Values below rmin will be extrapolated as the first tabulated value."
        )

    # ------------------------------------------------------------
    # LAMMPS grids
    # ------------------------------------------------------------
    # EAM format uses:
    #   rho grid: 0, drho, 2drho, ...
    #   r grid:   0, dr,   2dr, ...
    #
    # cutoff must be the last r value.
    # ------------------------------------------------------------
    rcut = rmax_export
    rhomax = rhomax_export

    dr = rcut / (NR - 1)
    drho = rhomax / (NRHO - 1)

    r_new = np.linspace(0.0, rcut, NR)
    rho_new = np.linspace(0.0, rhomax, NRHO)

    # ------------------------------------------------------------
    # Interpolate functions onto LAMMPS grids
    # ------------------------------------------------------------
    F_new = interp_to_grid(
        rho_old,
        F_old,
        rho_new,
        name="F(rho)",
    )

    rho_func_new = interp_to_grid(
        r_old,
        rho_func_old,
        r_new,
        name="rho(r)",
    )

    phi_new = interp_to_grid(
        r_old,
        phi_old,
        r_new,
        name="phi(r)",
    )

    # ------------------------------------------------------------
    # Clean and enforce physical tails
    # ------------------------------------------------------------
    if GAUGE_SHIFT_F_ZERO:
        F_new = F_new - F_new[0]

    if CLIP_RHO_NONNEGATIVE:
        rho_func_new = np.clip(rho_func_new, 0.0, None)

    if FORCE_TAIL_ZERO:
        rho_func_new[-1] = 0.0
        phi_new[-1] = 0.0

    # Avoid singular r*phi behavior at r=0.
    # LAMMPS eam/fs pair section stores r * phi(r), not phi(r).
    r_phi_new = r_new * phi_new
    r_phi_new[0] = 0.0

    F_new = clean_array(F_new, "F_new")
    rho_func_new = clean_array(rho_func_new, "rho_func_new")
    phi_new = clean_array(phi_new, "phi_new")
    r_phi_new = clean_array(r_phi_new, "r_phi_new")

    # ------------------------------------------------------------
    # Print diagnostics
    # ------------------------------------------------------------
    print("\n[INFO] Final table information")
    print(f"NRHO  = {NRHO}")
    print(f"drho  = {drho:.16E}")
    print(f"NR    = {NR}")
    print(f"dr    = {dr:.16E}")
    print(f"cutoff= {rcut:.16E}")
    print("")

    print_array_info("F(rho)", F_new)
    print_array_info("rho(r)", rho_func_new)
    print_array_info("phi(r)", phi_new)
    print_array_info("r*phi(r)", r_phi_new)

    # ------------------------------------------------------------
    # Write LAMMPS eam/fs file
    # ------------------------------------------------------------
    with OUTPUT_EAMFS.open("w") as f:
        # First three comment lines
        f.write("KAN-EAM Ag potential generated from trainable F, rho, phi\n")
        f.write(f"Source npz: {INPUT_NPZ}\n")
        f.write("Single-element Ag eam/fs file; pair section stores r*phi(r)\n")

        # Element line
        f.write(f"1  {ELEMENT}\n")

        # Global grid line
        f.write(
            f"{NRHO:5d} {drho:24.16E} "
            f"{NR:5d} {dr:24.16E} "
            f"{rcut:24.16E}\n"
        )

        # Element header
        f.write(
            f"{Z:5d} {MASS:24.16E} "
            f"{LATTICE_CONSTANT:24.16E} {LATTICE_TYPE}\n"
        )

        # Embedding function F(rho)
        write_values(f, F_new, values_per_line=VALUES_PER_LINE)

        # Electron density function rho_Ag->Ag(r)
        # For single-element eam/fs, only one density table is needed.
        write_values(f, rho_func_new, values_per_line=VALUES_PER_LINE)

        # Pair potential section: r * phi_AgAg(r)
        write_values(f, r_phi_new, values_per_line=VALUES_PER_LINE)

    print("\n============================================================")
    print("[DONE] Wrote LAMMPS eam/fs potential file")
    print(f"[DONE] {OUTPUT_EAMFS}")
    print("============================================================")

    print("\nUse in LAMMPS:")
    print("------------------------------------------------------------")
    print("pair_style  eam/fs")
    print(f"pair_coeff  * * {OUTPUT_EAMFS.name} Ag")
    print("------------------------------------------------------------")


if __name__ == "__main__":
    main()