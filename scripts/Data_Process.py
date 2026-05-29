"""
data_process.py

Convert VASP OUTCAR files into training data for KAN-EAM/MEAM
machine-learning interatomic potential training.

Expected input:
    OUTCAR files from VASP calculations

Expected output:
    Processed training data containing structures, energies, forces,
    cell information, and atomic species.

Author:
    Huang, Hung-Liang
"""





from pathlib import Path
import re
import json
import numpy as np


# ============================================================
# User settings
# ============================================================

# Root folders containing completed VASP calculations
INPUT_ROOTS = [
    Path("01_anchor_fcc_eos"),
    Path("02_anchor_fcc_small_displacement"),
    Path("03_anchor_fcc_elastic_strain"),
]

# Output folder for KAN-EAM training
OUTPUT_ROOT = Path("training_data_Ag_stage1")

# Element used if POSCAR does not explicitly contain element symbols
DEFAULT_ELEMENT = "Ag"

# If True, skip folders that do not appear to have finished normally
# If your jobs were killed or incomplete, keep this True.
REQUIRE_NORMAL_VASP_END = True

# If True, overwrite existing .npz files
OVERWRITE = True

# VASP OUTCAR filename
OUTCAR_NAME = "OUTCAR"

# POSCAR filename
POSCAR_NAME = "POSCAR"


# ============================================================
# Basic element table
# Extend this if needed later.
# ============================================================

ELEMENT_Z = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ag": 47,
    "Au": 79,
}


# ============================================================
# POSCAR parser
# ============================================================

def is_int_list(tokens):
    try:
        [int(x) for x in tokens]
        return True
    except Exception:
        return False


def read_poscar_species_and_counts(poscar_path: Path):
    """
    Read element symbols and counts from POSCAR.

    Supports normal VASP 5 format:

        Ag
        4

    Also supports older VASP style without element symbols by using DEFAULT_ELEMENT.
    """
    lines = poscar_path.read_text(errors="ignore").splitlines()

    if len(lines) < 8:
        raise ValueError(f"POSCAR is too short: {poscar_path}")

    line5 = lines[5].split()
    line6 = lines[6].split()

    # VASP 4 style: line 5 is atom counts
    if is_int_list(line5):
        elements = [DEFAULT_ELEMENT]
        counts = [int(x) for x in line5]
    else:
        elements = line5
        counts = [int(x) for x in line6]

    if len(elements) != len(counts):
        raise ValueError(
            f"Element/count mismatch in POSCAR: {poscar_path}\n"
            f"elements = {elements}\n"
            f"counts   = {counts}"
        )

    numbers = []
    for elem, count in zip(elements, counts):
        if elem not in ELEMENT_Z:
            raise ValueError(
                f"Unknown element symbol '{elem}' in {poscar_path}. "
                f"Please add it to ELEMENT_Z."
            )
        numbers.extend([ELEMENT_Z[elem]] * count)

    return elements, counts, np.array(numbers, dtype=np.int32)


# ============================================================
# OUTCAR parser
# ============================================================

_float_pattern = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][-+]?\d+)?"


def outcar_finished_normally(outcar_path: Path):
    """
    Detect whether VASP finished normally.

    VASP usually prints:
        General timing and accounting informations for this job:
    near the end of OUTCAR.
    """
    try:
        text_tail = outcar_path.read_text(errors="ignore")[-20000:]
    except Exception:
        return False

    markers = [
        "General timing and accounting informations for this job",
        "Voluntary context switches",
    ]

    return any(m in text_tail for m in markers)


def parse_outcar(outcar_path: Path, natoms_expected: int):
    """
    Extract final energy, final cell, final positions, and final forces from OUTCAR.

    Returns:
        cell: (3,3) ndarray, Angstrom
        positions: (N,3) ndarray, Angstrom
        forces: (N,3) ndarray, eV/Angstrom
        energy: float, eV
    """

    last_energy = None
    last_cell = None
    last_positions = None
    last_forces = None

    lines = outcar_path.read_text(errors="ignore").splitlines()
    n_lines = len(lines)

    i = 0
    while i < n_lines:
        line = lines[i]

        # ------------------------------------------------------------
        # Final total energy
        # Example:
        # free  energy   TOTEN  =      -11.23456789 eV
        # ------------------------------------------------------------
        if "free  energy   TOTEN" in line:
            match = re.search(r"TOTEN\s+=\s+(" + _float_pattern + r")", line)
            if match:
                last_energy = float(match.group(1))

        # ------------------------------------------------------------
        # Cell vectors
        # OUTCAR block:
        # direct lattice vectors                 reciprocal lattice vectors
        #    ax ay az ...
        #    bx by bz ...
        #    cx cy cz ...
        # ------------------------------------------------------------
        if "direct lattice vectors" in line and "reciprocal lattice vectors" in line:
            if i + 3 < n_lines:
                try:
                    cell = []
                    for j in range(1, 4):
                        vals = lines[i + j].split()
                        cell.append([float(vals[0]), float(vals[1]), float(vals[2])])
                    last_cell = np.array(cell, dtype=np.float64)
                except Exception:
                    pass

        # ------------------------------------------------------------
        # Positions and forces
        # OUTCAR block:
        # POSITION                                       TOTAL-FORCE (eV/Angst)
        # -----------------------------------------------------------------------------------
        # x y z fx fy fz
        # ------------------------------------------------------------
        if "POSITION" in line and "TOTAL-FORCE" in line:
            # Usually next line is dashed line, data starts at i+2
            start = i + 2
            positions = []
            forces = []

            for j in range(start, min(start + natoms_expected, n_lines)):
                vals = lines[j].split()
                if len(vals) < 6:
                    break

                try:
                    x, y, z = float(vals[0]), float(vals[1]), float(vals[2])
                    fx, fy, fz = float(vals[3]), float(vals[4]), float(vals[5])
                except Exception:
                    break

                positions.append([x, y, z])
                forces.append([fx, fy, fz])

            if len(positions) == natoms_expected:
                last_positions = np.array(positions, dtype=np.float64)
                last_forces = np.array(forces, dtype=np.float64)

        i += 1

    missing = []
    if last_energy is None:
        missing.append("energy")
    if last_cell is None:
        missing.append("cell")
    if last_positions is None:
        missing.append("positions")
    if last_forces is None:
        missing.append("forces")

    if missing:
        raise ValueError(
            f"Could not parse {missing} from OUTCAR: {outcar_path}"
        )

    return last_cell, last_positions, last_forces, float(last_energy)


# ============================================================
# Conversion
# ============================================================

def classify_group(path: Path):
    """
    Label the source group based on folder name.
    This is stored only as metadata.
    """
    p = str(path).replace("\\", "/").lower()

    if "01_anchor_fcc_eos" in p:
        return "01_anchor_fcc_eos"
    if "02_anchor_fcc_small_displacement" in p:
        return "02_anchor_fcc_small_displacement"
    if "03_anchor_fcc_elastic_strain" in p:
        return "03_anchor_fcc_elastic_strain"

    return "unknown"


def make_output_path(calc_dir: Path, input_root: Path):
    """
    Mirror the input directory structure into OUTPUT_ROOT.

    Example:
        02_anchor_fcc_small_displacement/disp_0.020A/structure_001/OUTCAR

    becomes:
        training_data_Ag_stage1/02_anchor_fcc_small_displacement/disp_0.020A/structure_001/frame.npz
    """
    relative = calc_dir.relative_to(input_root)
    out_dir = OUTPUT_ROOT / input_root.name / relative
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "frame.npz"


def convert_one_calculation(calc_dir: Path, input_root: Path):
    outcar_path = calc_dir / OUTCAR_NAME
    poscar_path = calc_dir / POSCAR_NAME

    if not outcar_path.exists():
        raise FileNotFoundError(f"Missing OUTCAR: {outcar_path}")

    if not poscar_path.exists():
        raise FileNotFoundError(f"Missing POSCAR: {poscar_path}")

    if REQUIRE_NORMAL_VASP_END and not outcar_finished_normally(outcar_path):
        raise RuntimeError(f"OUTCAR does not appear to have finished normally: {outcar_path}")

    elements, counts, numbers = read_poscar_species_and_counts(poscar_path)
    natoms = len(numbers)

    cell, positions, forces, energy = parse_outcar(
        outcar_path=outcar_path,
        natoms_expected=natoms,
    )

    if positions.shape != forces.shape:
        raise ValueError(f"positions and forces shape mismatch in {calc_dir}")

    if positions.shape[0] != natoms:
        raise ValueError(
            f"Atom number mismatch in {calc_dir}: "
            f"POSCAR has {natoms}, OUTCAR has {positions.shape[0]}"
        )

    pbc = np.array([True, True, True], dtype=np.bool_)

    out_npz = make_output_path(calc_dir, input_root)

    if out_npz.exists() and not OVERWRITE:
        return {
            "status": "skipped_exists",
            "calc_dir": str(calc_dir),
            "out_npz": str(out_npz),
            "natoms": natoms,
            "energy": energy,
        }

    group = classify_group(calc_dir)

    # Main keys expected by your KAN-EAM code:
    # cell, pbc, positions, numbers, energy, forces
    np.savez_compressed(
        out_npz,
        cell=cell.astype(np.float64),
        pbc=pbc,
        positions=positions.astype(np.float64),
        numbers=numbers.astype(np.int32),
        energy=np.array(energy, dtype=np.float64),
        forces=forces.astype(np.float64),

        # Extra metadata, harmless for the training code
        elements=np.array(elements),
        counts=np.array(counts, dtype=np.int32),
        source_dir=np.array(str(calc_dir)),
        source_outcar=np.array(str(outcar_path)),
        source_poscar=np.array(str(poscar_path)),
        group=np.array(group),
    )

    return {
        "status": "converted",
        "calc_dir": str(calc_dir),
        "out_npz": str(out_npz),
        "group": group,
        "natoms": natoms,
        "energy": energy,
        "max_force": float(np.max(np.linalg.norm(forces, axis=1))),
    }


def find_calculation_dirs(input_root: Path):
    """
    Find all directories under input_root that contain OUTCAR.
    """
    outcars = sorted(input_root.rglob(OUTCAR_NAME))
    calc_dirs = [p.parent for p in outcars]
    return calc_dirs


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("============================================================")
    print(" Convert VASP OUTCAR files to KAN-EAM .npz database")
    print("============================================================")
    print(f"Output root: {OUTPUT_ROOT}")
    print(f"Require normal VASP end: {REQUIRE_NORMAL_VASP_END}")
    print("Input roots:")
    for root in INPUT_ROOTS:
        print(f"  - {root}")
    print("============================================================")

    all_results = []
    n_converted = 0
    n_failed = 0
    n_skipped = 0

    for input_root in INPUT_ROOTS:
        if not input_root.exists():
            print(f"[WARN] Missing input root, skip: {input_root}")
            continue

        calc_dirs = find_calculation_dirs(input_root)

        print(f"\n[INFO] {input_root}: found {len(calc_dirs)} OUTCAR files")

        for calc_dir in calc_dirs:
            try:
                result = convert_one_calculation(calc_dir, input_root)
                all_results.append(result)

                if result["status"] == "converted":
                    n_converted += 1
                    print(
                        f"[OK] {result['group']} | atoms={result['natoms']:5d} | "
                        f"E={result['energy']: .8f} eV | "
                        f"maxF={result['max_force']: .6f} eV/A | "
                        f"{result['out_npz']}"
                    )
                elif result["status"] == "skipped_exists":
                    n_skipped += 1
                    print(f"[SKIP] exists: {result['out_npz']}")

            except Exception as e:
                n_failed += 1
                fail_record = {
                    "status": "failed",
                    "calc_dir": str(calc_dir),
                    "error": str(e),
                }
                all_results.append(fail_record)
                print(f"[FAIL] {calc_dir}")
                print(f"       reason: {e}")

    # Save conversion report
    report_path = OUTPUT_ROOT / "conversion_report.json"
    with report_path.open("w") as f:
        json.dump(all_results, f, indent=2)

    # Save simple text summary
    summary_path = OUTPUT_ROOT / "conversion_summary.txt"
    with summary_path.open("w") as f:
        f.write("VASP OUTCAR to KAN-EAM npz conversion summary\n")
        f.write("============================================================\n")
        f.write(f"Converted: {n_converted}\n")
        f.write(f"Skipped:   {n_skipped}\n")
        f.write(f"Failed:    {n_failed}\n")
        f.write(f"Output:    {OUTPUT_ROOT}\n")
        f.write(f"Report:    {report_path}\n")

    print("\n============================================================")
    print("[DONE] Conversion finished.")
    print(f"[DONE] Converted: {n_converted}")
    print(f"[DONE] Skipped:   {n_skipped}")
    print(f"[DONE] Failed:    {n_failed}")
    print(f"[DONE] Report:    {report_path}")
    print(f"[DONE] Summary:   {summary_path}")
    print("============================================================")


if __name__ == "__main__":
    main()