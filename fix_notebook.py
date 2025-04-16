#!/usr/bin/env python3
"""
This script modifies the 1_end2end_5lines.ipynb notebook to fix tensor dimension issues
and update the DIV2K dataset path to /nas1/DIV2K.
"""
import json
import os
import copy

NOTEBOOK_PATH = "/home/yz/DeepLens/1_end2end_5lines.ipynb"

# First, read the notebook
with open(NOTEBOOK_PATH, 'r') as f:
    notebook = json.load(f)

# Create backup
backup_path = NOTEBOOK_PATH + ".bak"
with open(backup_path, 'w') as f:
    json.dump(notebook, f, indent=2)
print(f"Created backup at {backup_path}")

# Import the safe_denormalize function
import_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "source": [
        "# Fix for tensor dimension issues\n",
        "from fix_denormalize import safe_denormalize, vis_sample"
    ],
    "outputs": []
}

# Cell to modify the DIV2K dataset path
dataset_path_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "source": [
        "# Update DIV2K dataset path\n",
        "DIV2K_PATH = '/nas1/DIV2K'\n",
        "os.makedirs(DIV2K_PATH, exist_ok=True)\n",
        "print(f'DIV2K dataset will be stored in {DIV2K_PATH}')"
    ],
    "outputs": []
}

# For end2end_train function - find it and modify it
modified_cells = []
for cell in notebook['cells']:
    if cell['cell_type'] == 'code' and 'def end2end_train' in ''.join(cell['source']):
        # Modify the end2end_train function to use safe_denormalize and the new dataset path
        new_source = []
        for line in cell['source']:
            if 'DIV2K_PATH' in line and '/nas1/DIV2K' not in line:
                new_source.append("    DIV2K_PATH = '/nas1/DIV2K'  # Updated path\n")
            elif 'denormalize_ImageNet' in line:
                new_source.append(line.replace('denormalize_ImageNet', 'safe_denormalize'))
            else:
                new_source.append(line)
        cell['source'] = new_source
    
    modified_cells.append(cell)

# Insert our import and dataset path cells near the beginning
notebook['cells'] = ([notebook['cells'][0]] + [import_cell, dataset_path_cell] + 
                     notebook['cells'][1:])

# Write the modified notebook
with open(NOTEBOOK_PATH, 'w') as f:
    json.dump(notebook, f, indent=2)

print(f"Modified notebook {NOTEBOOK_PATH} to use safe_denormalize function and update DIV2K path")
print("Please run the notebook and use the vis_sample function instead of manual visualization code")
print("Example usage of vis_sample:")
print("```")
print("vis_sample(img_org, img_render, img_rec, loss=loss.item(), epoch=epoch, batch=batch_idx)")
print("```") 