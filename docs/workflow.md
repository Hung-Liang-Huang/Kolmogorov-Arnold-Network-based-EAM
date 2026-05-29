# Workflow

This repository follows the workflow below:

```text
VASP OUTCAR files
        ↓
data_process.py
        ↓
processed training data
        ↓
training.py
        ↓
trained PyTorch model
        ↓
validate.py
        ↓
energy and force parity plots
        ↓
export_lammps.py
        ↓
LAMMPS-compatible EAM/fs potential
```

## Step 1: VASP Data Processing

The script `scripts/data_process.py` reads VASP OUTCAR files and converts them into training data.

Expected input:

```text
OUTCAR files from VASP calculations
```

Expected output:

```text
processed training data containing structures, total energies, atomic forces, cell information, and atomic species
```

The full VASP datasets are not included in this repository because they may contain unpublished or collaborator-owned research data.

## Step 2: Model Training

The script `scripts/training.py` trains the KAN-EAM/MEAM model using the processed DFT data.

Typical outputs include:

```text
trained PyTorch model checkpoint
training loss history
energy and force errors
```

## Step 3: Validation

The script `scripts/validate.py` evaluates the trained model by comparing predicted energies and forces with DFT reference values.

Typical outputs include:

```text
energy parity plot
force parity plot
energy RMSE
force RMSE
```

## Step 4: LAMMPS Export

The script `scripts/export_lammps.py` converts the trained `.pt` model into a LAMMPS-compatible EAM/fs potential file.

Typical output:

```text
KAN_model.eam.fs
```

## Step 5: LAMMPS Testing

The exported potential can be tested using the input templates in `lammps_examples/`.

Recommended checks include:

- whether LAMMPS can read the exported potential,
- structural minimization,
- energy consistency between PyTorch and LAMMPS,
- pressure after minimization,
- short NVT or NVE molecular dynamics stability tests.