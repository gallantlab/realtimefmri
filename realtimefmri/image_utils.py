import os
import os.path as op
import shlex
import tempfile
import subprocess

import nibabel
import numpy as np

import cortex
from realtimefmri import utils

logger = utils.get_logger('image_utils', to_console=True)


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
           '-f', 'mystudy%s',
           '-o', d.name, dicom_path]

    _ = utils.run_command(cmd, stdout=subprocess.DEVNULL)
    nii = nibabel.load(op.join(d.name, os.listdir(d.name)[0]), mmap=False)

    affine = nii.affine
    affine[1, 1] *= -1
    nii = nibabel.Nifti1Image(nii.get_data()[:, ::-1], affine, nii.header)

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


def register(volume, reference, twopass=False, output_transform=False):
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
        reference_path = utils.get_temporary_path(directory=temp_directory.name,
                                                  extension='.nii.gz')
        nibabel.save(reference, reference_path)

    if isinstance(volume, str):
        volume_path = volume
    else:
        volume_path = utils.get_temporary_path(directory=temp_directory.name, extension='.nii.gz')
        nibabel.save(volume, volume_path)

    registered_volume_path = utils.get_temporary_path(directory=temp_directory.name,
                                                      extension='.nii')
    cmd = shlex.split('3dvolreg -base {} -prefix {}'.format(reference_path,
                                                            registered_volume_path))
    if output_transform:
        transform_path = utils.get_temporary_path(temp_directory.name, extension='.aff12.1D')
        cmd.extend(['-1Dmatrix_save', transform_path])
    if twopass:
        cmd.append('-twopass')

    cmd.append(volume_path)

    env = os.environ.copy()
    env['AFNI_NIFTI_TYPE_WARN'] = 'NO'
    error_message = utils.run_command(cmd, raise_errors=False, env=env)
    if error_message is not None:
        logger.debug(error_message)

    registered_volume = nibabel.load(registered_volume_path)
    registered_volume.get_data()

    if output_transform:
        xfm = load_afni_xfm(transform_path)
        return registered_volume, xfm

    else:
        return registered_volume


def mosaic_to_volume(mosaic, nrows=6, ncols=6):
    volume = np.empty((100, 100, nrows * ncols))
    for i in range(nrows):
        vol = mosaic[i * 100:(i + 1) * 100, :].reshape(100, 100, ncols, order='F')
        volume[:, :, i * ncols:(i + 1) * ncols] = vol
    return volume


def decompose_affine(affine):
    """Decompose a affine matrix into pitch, roll, and yaw, x, y, z displacement components

    References
    ----------
    .. [1] Planning Algorithms, Steven M. LaValle. 3.2.3 3D Transformations. Cambridge University
           Press <http://planning.cs.uiuc.edu/node103.html>
    """
    alpha = np.arctan2(affine[1, 0], affine[0, 0])
    beta = np.arctan2(-affine[2, 0], np.sqrt(affine[2, 1]**2 + affine[2, 2]**2))
    gamma = np.arctan2(affine[2, 1], affine[2, 2])

    x, y, z = affine[:3, -1]

    return alpha, beta, gamma, x, y, z


def load_afni_xfm(path):
    return np.r_[np.loadtxt(path).reshape(3, 4), np.array([[0, 0, 0, 1]])]


def load_mask(surface, transform, mask_type):
    mask_path = op.join(cortex.database.default_filestore,
                        surface, 'transforms', transform,
                        'mask_' + mask_type + '.nii.gz')
    return nibabel.load(mask_path)


def load_reference(surface, transform):
    ref_path = op.join(cortex.database.default_filestore,
                       surface, 'transforms', transform,
                       'reference.nii.gz')
    return nibabel.load(ref_path)
