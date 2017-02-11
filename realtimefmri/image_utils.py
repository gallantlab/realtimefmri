import os
import os.path as op
from subprocess import call, STDOUT
import shlex
from tempfile import gettempdir
tempdir = gettempdir()

import numpy as np
from uuid import uuid4
from nibabel import load as nbload, save as nbsave

def transform(inp, base, output_transform=False):
    if isinstance(base, basestring):
        base_path = base
    else:
        base_path = op.join(tempdir, str(uuid4())+'.nii')
        nbsave(base, base_path)

    if isinstance(inp, basestring):
        inp_path = inp
    else:
        inp_path = op.join(tempdir, str(uuid4())+'.nii')
        nbsave(inp, inp_path)

    out_path = op.join(tempdir, str(uuid4())+'.nii')
    cmd = shlex.split('3dvolreg -base {} -prefix {}'.format(base_path, out_path))
    if output_transform:
        transform_path = op.join(tempdir, str(uuid4())+'.aff12.1D')
        cmd.append(['-1Dmatrix_save', transform_path])        
    cmd.append(inp_path)
    
    devnull = open(os.devnull, 'w')
    ret = call(cmd, stdout=devnull, stderr=STDOUT, close_fds=True)
    if ret>0:
        print ' '.join(cmd)

    out_img = nbload(out_path)
    out_img.get_data()

    os.remove(out_path)
    if inp is not inp_path:
        os.remove(inp_path)
    if base is not base_path:
        os.remove(base_path)

    if output_transform:
        xfm = load_afni_xfm(transform_path)
        os.remove(transform_path)
        return out_img, xfm    
        
    else:
        return out_img

def mosaic_to_volume(mosaic, nrows=6, ncols=6):
    volume = np.empty((100,100, nrows*ncols))
    for i in xrange(nrows):
        volume[:,:,i*ncols:(i+1)*ncols] = mosaic[i*100:(i+1)*100, :].reshape(100,100,ncols,order='F')
    return volume

def plot_volume(volume):
    from matplotlib import pyplot as plt
    nslices = volume.shape[2]
    nrows = ncols = np.ceil(nslices**0.5).astype(int)
    
    fig, ax = plt.subplots(nrows, ncols)
    for i in xrange(volume.shape[2]):
        ax[divmod(i,ncols)].pcolormesh(volume[:,:,i], cmap='gray');

def load_afni_xfm(path):
    return np.r_[np.loadtxt(path).reshape(3,4), np.array([[0,0,0,1]])]

