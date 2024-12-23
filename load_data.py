import os
import glob
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split


def load_data(bs):

    """
    Loads input and output data, splits into training and testing sets, and returns DataLoaders.

    Args:
        bs batch_size (int): Batch size for the DataLoader.

    Returns:
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validating set.
        test_loader (DataLoader): DataLoader for the testing set.
    """

    # Set Directories
    base_dir = '../ML_ion_conc_3D_finegrid/training_input/'
    dir_conc = os.path.join(base_dir, 'conc_downsample2/')
    dir_charge = os.path.join(base_dir, 'charge_downsample2/')
    dir_rad = os.path.join(base_dir, 'vdw_downsample2/')

    # File matching pattern
    files1 = glob.glob(os.path.join(dir_conc, "D_29*.dx.npy"))

    # Initialize data containers
    output_data = []
    input_data = []

    # Open output files for logging
    outputfile1 = open('loadfile.txt', 'w')
    outputfile2 = open('test_index.txt','w')

    print("############ Loading data ###########")
    for i, file1 in enumerate(files1):
        basename = os.path.basename(file1)
        print(f"{i} {basename}", file=outputfile1)

        # Load concentration data
        tmp_conc = np.expand_dims(np.load(file1), axis=0)

        # Construct and load radial and charge data
        base_name = basename.rsplit('_ndens.dx.npy', 1)[0]
        tmp_rad = np.load(os.path.join(dir_rad, f"{base_name}_vdw.dx.npy"))
        tmp_charge = np.load(os.path.join(dir_charge, f"{base_name}_charge.dx.npy"))

        # Combine inputs and append to lists
        tmp_input = np.stack((tmp_rad, tmp_charge), axis=0)
        output_data.append(tmp_conc)
        input_data.append(tmp_input)

    # Convert data to PyTorch tensors
    input_data = torch.from_numpy(np.array(input_data))
    output_data = torch.from_numpy(np.array(output_data))


    # Split data into training, validating and testing sets
    train_input, temp_input, train_output, temp_output, _, _ = train_test_split(
        input_data, output_data, np.arange(len(input_data)), test_size=0.2
    )

    val_input, test_input, val_output, test_output, _, test_indices = train_test_split(
        temp_input, temp_output, np.arange(len(temp_input)), test_size=0.5
    )

    # Log test indices
    print("test_indices", file=outputfile2)
    print(test_indices, file=outputfile2)

    # Create TensorDatasets
    train_dataset = TensorDataset(train_input, train_output)
    val_dataset = TensorDataset(val_input, val_output)
    test_dataset = TensorDataset(test_input, test_output)


    # Create DataLoader for training
    train_loader = DataLoader(
        train_dataset, batch_size=bs, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=bs, shuffle=True, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=bs, shuffle=True, num_workers=2, pin_memory=True
    )

    return train_loader, val_loader, test_loader

