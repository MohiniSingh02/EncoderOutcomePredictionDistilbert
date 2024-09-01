import logging
import os
import sys
import torch


def rename_ckpt_files(directory):
    # Walk through the directory, including all subdirectories
    for root, _, files in os.walk(directory):
        for filename in files:
            # Check if the file ends with '.ckpt'
            if filename.endswith('.ckpt'):
                path = os.path.join(root, filename)
                ckpt = torch.load(path)
                ckpt['hyper_parameters']['num_classes'] = len(ckpt['state_dict']['thresholds'])
                torch.save(ckpt, path)
                logging.warning(f"Fixed: {filename} -> {ckpt['hyper_parameters']['num_classes']}")


if __name__ == "__main__":
    # Check if the directory argument is provided
    if len(sys.argv) != 2:
        print("Usage: python rename_ckpt_files.py <directory>")
        sys.exit(1)

    # Get the directory from the command-line argument
    directory = sys.argv[1]

    # Check if the provided path is a directory
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory")
        sys.exit(1)

    # Call the function to rename files
    rename_ckpt_files(directory)
