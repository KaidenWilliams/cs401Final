# Code for Step 4: Package the Model. Zip the model into a tar with the model_dependencies folder (inference.py and requirements.txt)


import os
import tarfile
import argparse
import glob
from pathlib import Path


def package_model():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default='/opt/ml/processing/model')
    parser.add_argument('--output-dir', type=str, default='/opt/ml/processing/output')
    parser.add_argument('--code-dir', type=str, default='/opt/ml/processing/input/dependencies')
    args = parser.parse_args()

    print(f"Received arguments: {args}")

    model_dir = args.model_dir
    output_dir = args.output_dir
    code_dir = args.code_dir

    os.makedirs(output_dir, exist_ok=True)

    model_dir_path = Path(model_dir)
    for file_path in model_dir_path.rglob('*'):
        if file_path.is_file():
            print(f"file in model_dir: {file_path}")

    # Apparently previous TrainingStep already zips model, so we need to unzip it to add inference.py and requirements.txt
    input_tar_files = glob.glob(os.path.join(model_dir, '*.tar.gz'))
    if len(input_tar_files) == 1:
        file = input_tar_files[0]
        with tarfile.open(file, "r:gz") as tar:
            tar.extractall(path=model_dir)

    script_file = os.path.join(code_dir, "inference.py")
    req_file = os.path.join(code_dir, "requirements.txt")

    weights_file = os.path.join(model_dir, "best_model.pth")
    map_file = os.path.join(model_dir, "species_mapping.json")

    files_to_package = {
        script_file: "code/inference.py",
        req_file: "code/requirements.txt",
        weights_file: "best_model.pth",
        map_file: "species_mapping.json",
    }

    tar_path = os.path.join(output_dir, "model.tar.gz")
    print(f"Creating tar archive at {tar_path}...")

    with tarfile.open(tar_path, "w:gz") as tar:
        for src_path, arc_name in files_to_package.items():
            if os.path.exists(src_path):
                print(f"Adding {src_path} as {arc_name}")
                tar.add(src_path, arcname=arc_name)
            else:
                print(f"Source file not found, skipping: {src_path}")


# Entry Point
if __name__ == "__main__":
    package_model()