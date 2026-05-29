@echo off

echo Step 1: Processing VASP OUTCAR files
python ..\..\scripts\data_process.py

echo Step 2: Training KAN-EAM/MEAM model
python ..\..\scripts\training.py

echo Step 3: Validating trained model
python ..\..\scripts\validate.py

echo Step 4: Exporting trained model to LAMMPS EAM/fs format
python ..\..\scripts\export_lammps.py

echo Workflow finished.
pause