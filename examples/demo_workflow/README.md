# Demo Workflow

This folder describes an example workflow for using the KAN-EAM/MEAM potential framework.

The full DFT datasets are not included because they may contain unpublished or collaborator-owned research data. This folder therefore provides the expected command sequence and configuration format.

## Workflow

1. Prepare VASP OUTCAR files.

2. Convert OUTCAR files into training data:

```bash
python ../../scripts/data_process.py
```

3. Train the model:

```bash
python ../../scripts/training.py
```

4. Validate the trained model:

```bash
python ../../scripts/validate.py
```

5. Export the trained model to a LAMMPS-compatible EAM/fs file:

```bash
python ../../scripts/export_lammps.py
```

## Notes

The scripts may require editing file paths depending on the local directory structure.

Private datasets, trained checkpoints, and unpublished collaborator data are not included in this repository.