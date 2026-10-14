import os
import sys
from pathlib import Path
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

WORKSPACE = Path.cwd()
CHALLENGE_DIR = WORKSPACE / "challenge"

notebooks_to_run = [
    CHALLENGE_DIR / "01_data_challenge.ipynb",
    CHALLENGE_DIR / "02_training_challenge.ipynb",
    CHALLENGE_DIR / "03_evaluation_deployment_challenge.ipynb",
]

ep = ExecutePreprocessor(timeout=600, kernel_name="python3")

for nb_path in notebooks_to_run:
    print(f"Ejecutando notebook: {nb_path.name}...")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    
    # Execute in the challenge directory so relative paths work perfectly
    ep.preprocess(nb, {"metadata": {"path": str(CHALLENGE_DIR)}})
    
    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"OK: {nb_path.name} ejecutado y guardado con sus salidas.\n")

print("Todos los notebooks fueron ejecutados exitosamente!")
