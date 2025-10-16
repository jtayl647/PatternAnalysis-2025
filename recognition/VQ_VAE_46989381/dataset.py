# Containing the data loader for loading and preprocessing your data.

import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

def to_channels(arr:np.ndarray , dtype = np.uint8 ) -> np.ndarray :
    channels = np.unique(arr)
    res = np.zeros (arr.shape + (len(channels),), dtype = dtype)
    for c in channels :
        c = int(c)
        res [..., c:c+1][arr == c] = 1
    return res

# load medical image functions
def load_data_2D (imageNames, normImage = False, categorical = False, dtype = np.float32 , 
                  getAffines = False, early_stop = False) :
    '''
        Load medical image data from names , cases list provided into a list for each.
        This function pre - allocates 4D arrays for conv2d to avoid excessive memory usage.
        
        normImage  : bool (normalise the image 0.0 -1.0)
        early_stop : Stop loading pre - maturely , leaves arrays mostly empty 
                     , for quick loading and testing scripts .
    '''
    affines = []

    # get fixed size
    num = len(imageNames)
    first_case = nib.load(imageNames[0]).get_fdata(caching = 'unchanged')
    if len (first_case.shape) == 3:
        first_case = first_case [: ,: ,0] # sometimes extra dims , remove
    if categorical :
        first_case = to_channels (first_case , dtype = dtype)
        rows, cols, channels = first_case.shape
        images = np.zeros ((num, rows, cols, channels), dtype = dtype)
    else :
        rows, cols = first_case.shape
        images = np.zeros ((num, rows, cols) , dtype = dtype)

    for i, inName in enumerate (imageNames):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching = 'unchanged') # read disk only
        affine = niftiImage.affine
        if len(inImage.shape) == 3:
            inImage = inImage [:,:,0] # sometimes extra dims in HipMRI_study data
        inImage = inImage.astype(dtype)
        if normImage:
            # ~ inImage = inImage / np . linalg . norm ( inImage )
            # ~ inImage = 255. * inImage / inImage . max ()
            inImage = (inImage - inImage.mean())/inImage.std()

        # Skip image if shape doesn't match first_case
        if inImage.shape[:2] != first_case.shape:
            continue  # move to next image

        if categorical:
            inImage = to_channels(inImage, dtype = dtype)
            images [i,:,:,:] = inImage
        else:
            images [i,:,:] = inImage

        affines.append(affine)
        if i > 20 and early_stop:
            break
    if getAffines:
        return images, affines
    else:
        return images

def load_our_data(base_path, normImage=False, batch_size=32):
    train_path = os.path.join(base_path, "keras_slices_train")
    val_path   = os.path.join(base_path, "keras_slices_validate")
    test_path  = os.path.join(base_path, "keras_slices_test")

    # Collect file paths
    train_files = sorted([os.path.join(train_path, f) for f in os.listdir(train_path) if f.endswith(".nii.gz")])
    val_files   = sorted([os.path.join(val_path, f)   for f in os.listdir(val_path)   if f.endswith(".nii.gz")])
    test_files  = sorted([os.path.join(test_path, f)  for f in os.listdir(test_path)  if f.endswith(".nii.gz")])

    # Load images
    X_train = load_data_2D(train_files, normImage=normImage)
    X_val   = load_data_2D(val_files, normImage=normImage)
    X_test  = load_data_2D(test_files, normImage=normImage)

    # Convert to tensors and add channel dim
    X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    X_val_t   = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
    X_test_t  = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)

    # DataLoaders
    train_loader = DataLoader(TensorDataset(X_train_t, X_train_t), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val_t, X_val_t), batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(TensorDataset(X_test_t, X_test_t), batch_size=batch_size, shuffle=False)

    # Print summary of shapes
    print("All tensors loaded successfully!")
    print(f"Train tensor shape: {X_train_t.shape}")
    print(f"Validation tensor shape: {X_val_t.shape}")
    print(f"Test tensor shape: {X_test_t.shape}")

    return train_loader, val_loader, test_loader, X_train_t