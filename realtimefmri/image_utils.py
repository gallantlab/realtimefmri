import six
if six.PY2:
    range = xrange
import os
import os.path as op
from shutil import rmtree
from subprocess import call, STDOUT
import shlex
from uuid import uuid4
from tempfile import mkdtemp, mkstemp
import numpy as np

import nibabel as nib
import dcmstack

def register(inp, base, output_transform=False, twopass=False):
    """Register the input image to the base image"""
    temp_directory = mkdtemp()

    try:
        if isinstance(base, six.string_types):
            base_path = base
        else:
            base_path = op.join(temp_directory, str(uuid4())+'.nii.gz')
            nib.save(base, base_path)

        if isinstance(inp, six.string_types):
            inp_path = inp
        else:
            inp_path = op.join(temp_directory, str(uuid4())+'.nii.gz')
            nib.save(inp, inp_path)

        out_path = op.join(temp_directory, str(uuid4())+'.nii')
        cmd = shlex.split('3dvolreg -base {} -prefix {}'.format(base_path,
                                                                out_path))
        if output_transform:
            transform_path = op.join(temp_directory, str(uuid4())+'.aff12.1D')
            cmd.extend(['-1Dmatrix_save', transform_path])
        if twopass:
            cmd.append('-twopass')

        cmd.append(inp_path)

        devnull = open(os.devnull, 'w')
        ret = call(cmd, stdout=devnull, stderr=STDOUT, close_fds=True)
        if ret > 0:
            print(' '.join(cmd))

        out_img = nib.load(out_path)
        out_img.get_data()

        if output_transform:
            xfm = load_afni_xfm(transform_path)

    except Exception as e:
        raise Exception(e)

    finally:
        rmtree(temp_directory)

    if output_transform:
        return out_img, xfm
    else:
        return out_img

def mosaic_to_volume(mosaic, nrows=6, ncols=6):
    volume = np.empty((100, 100, nrows*ncols))
    for i in range(nrows):
        vol = mosaic[i*100:(i+1)*100, :].reshape(100, 100, ncols, order='F')
        volume[:, :, i*ncols:(i+1)*ncols] = vol
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

def get_orientation_labels(target_codes):
    """Convert orientation in list-style to label-style

    Arguments:
    ----------
    target_codes : str
        String with three characters indicating the positive direction for each
        dimension, e.g., 'LPS'

    Returns:
    --------
    The orientation as a list of tuples, e.g. (('L', 'R'), ('P', 'A'), ('S', 'I'))
    """
    codes = (('L', 'R'), ('P', 'A'), ('S', 'I'))
    orientation_labels = []
    for code in codes:
        for co in target_codes:
            if co in code:
                if co == code[0]:
                    orientation_labels.append(code[::-1])
                else:
                    orientation_labels.append(code)
    return orientation_labels

def set_orientation(nii, target_orientation):
    """Set the orientation of a nifti volume

    Arguments:
    ----------
    nii : nibabel Nifti1 image
    target_orientation : list of tuples
        e.g., (('L', 'R'), ('P', 'A'), ('S', 'I'))
    """    
    orientation = nib.orientations.aff2axcodes(nii.affine)
    if orientation != target_orientation:
        data = nii.get_data()
        affine = nii.affine.copy()
        orientation_labels = get_orientation_labels(target_orientation)
        flip_axes = dcmstack.axcodes2ornt(orientation, orientation_labels)[:, 1]
        for i, flip in enumerate(flip_axes):
            if flip == -1:
                data = np.flip(data, i)
                affine[i, i] *= -1
        nii = nib.Nifti1Image(data, affine, header=nii.header)

    return nii
