import os
import os.path as op
import tempfile
import subprocess
import shlex
import numpy as np
import nibabel
from realtimefmri.utils import get_temporary_path


def dicom_to_nifti(dicom_path):
    """Convert dicom image to nibabel nifti

    Parameters
    ----------
    dicom_path : str
        Path to dicom image

    Returns
    -------
    A nibabel.nifti1.Nifti1Image
    """
    d = tempfile.TemporaryDirectory()
    cmd = ['dcm2niix',
           '-s', 'y',
           '-b', 'n',
           '-1',
           '-o', d.name, dicom_path]

    subprocess.check_call(cmd)
    nii = nibabel.load(op.join(d.name, os.listdir(d.name)[0]), mmap=False)
    _ = nii.get_data()
    d.cleanup()

    return nii


def secondary_mask(mask1, mask2, order='C'):
    """
    Given an array, X and two 3d masks, mask1 and mask2
    X[mask1] = x
    X[mask1 & mask2] = y
    x[new_mask] = y
    """
    assert mask1.shape == mask2.shape
    mask1_flat = mask1.ravel(order=order)
    mask2_flat = mask2.ravel(order=order)

    masks = np.c_[mask1_flat, mask2_flat]
    masks = masks[mask1_flat, :]
    return masks[:, 1].astype(bool)


def register(volume, reference, output_transform=False, twopass=False):
    """Register the input image to the reference image

    Parameters
    ----------
    volume : str or nibabel.nifti1.Nifti1Image
    reference : str or nibabel.nifti1.Nifti1Image
    output_transform : bool
    twopass : bool

    """
    temp_directory = tempfile.TemporaryDirectory()

    if isinstance(reference, str):
        reference_path = reference
    else:
        reference_path = get_temporary_path(directory=temp_directory.name, extension='.nii.gz')
        nibabel.save(reference, reference_path)

    if isinstance(volume, str):
        volume_path = volume
    else:
        volume_path = get_temporary_path(directory=temp_directory.name, extension='.nii.gz')
        nibabel.save(volume, volume_path)

    registered_volume_path = get_temporary_path(directory=temp_directory.name, extension='.nii')
    cmd = shlex.split('3dvolreg -base {} -prefix {}'.format(reference_path,
                                                            registered_volume_path))
    if output_transform:
        transform_path = get_temporary_path(temp_directory.name, extension='.aff12.1D')
        cmd.extend(['-1Dmatrix_save', transform_path])
    if twopass:
        cmd.append('-twopass')

    cmd.append(volume_path)

    with open(os.devnull, 'w') as devnull:
        ret = subprocess.call(cmd, stdout=devnull, stderr=subprocess.STDOUT, close_fds=True)

    if ret > 0:
        print(' '.join(cmd))

    registered_volume = nibabel.load(registered_volume_path)
    registered_volume.get_data()

    if output_transform:
        xfm = load_afni_xfm(transform_path)

    if output_transform:
        return registered_volume, xfm
    else:
        return registered_volume


def mosaic_to_volume(mosaic, nrows=6, ncols=6):
    volume = np.empty((100, 100, nrows * ncols))
    for i in range(nrows):
        vol = mosaic[i * 100:(i + 1) * 100, :].reshape(100, 100, ncols, order='F')
        volume[:, :, i * ncols:(i + 1) * ncols] = vol
    return volume


def plot_volume(volume):
    from matplotlib import pyplot as plt
    nslices = volume.shape[2]
    nrows = ncols = np.ceil(nslices**0.5).astype(int)

    _, ax = plt.subplots(nrows, ncols)
    for i in range(volume.shape[2]):
        _ = ax[divmod(i, ncols)].pcolormesh(volume[:, :, i], cmap='gray')


def load_afni_xfm(path):
    return np.r_[np.loadtxt(path).reshape(3, 4), np.array([[0, 0, 0, 1]])]
