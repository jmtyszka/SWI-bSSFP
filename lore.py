#!/usr/bin/env python3
"""
Pure Python implementation of

Usage
----
lore.py -i <4D bSSFP images>.nii.gz [-r <te/tr ratio>]

Performs recon of multiphase mag/phs bSSFP images (4D Nifti) and outputs separate LORE volumes (s0, etc)

Authors
----
Mike Tyszka, Caltech Brain Imaging Center

Dates
----
2017-02-07 JMT Implement from Bjork et al Magn Reson Med 72:880-892 (2014)

License
----
MIT License

Copyright (c) 2016 Mike Tyszka

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import sys
import argparse
import json
import nibabel as nib
import numpy as np
import multiprocessing as mp
from numpy import dot
from skimage.filters import threshold_otsu
from numpy.linalg import inv

__version__ = '1.0.0'


def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Multi-phase bSSFP LORE reconstruction')
    parser.add_argument('-i', '--infile', required=True, help='Input 4D bSSFP volumes in mag/phase format')
    parser.add_argument('-m', '--mask', default=False, type=bool, help='Use calculation mask [False]')
    parser.add_argument('-d', '--debug', default=0, type=int, help='Debug level [0]')

    # Parse command line arguments
    args = parser.parse_args()

    # Extract arguments
    bssfp_fname = args.infile
    use_mask = args.mask
    debug = args.debug

    # Load 4D mag/phase bSSFP volumes
    print('Loading bSSFP mag/phase volumes from %s' % bssfp_fname)
    bssfp_nii = nib.load(bssfp_fname)
    bssfp = bssfp_nii.get_data()

    # Load JSON information if available, otherwise use reasonable defaults
    json_fname = bssfp_fname.replace('.nii.gz', '.json').replace('.nii', '.json')

    if os.path.isfile(json_fname):

        print('Found JSON sidecare - loading sequence information')
        with open(json_fname) as json_data:
            info = json.load(json_data)

        tr, te, alpha = info['RepetitionTime'], info['EchoTime'], info['FlipAngle']

        # Convert from seconds to ms
        tr, te = tr * 1000.0, te * 1000.0

    else:

        print('No JSON sidecar found - using reasonable defaults')
        tr, te, alpha = 5.0, 2.5, 30.0

    print('(TR, TE, Flip) = (%0.2f ms, %0.2f ms, %0.1f deg)' % (tr, te, alpha))

    # Grab input image dimensions
    # 4th dimension contains nv re/im pairs (ie nt = 2*nv)
    nt = bssfp_nii.header.get_data_shape()[3]

    if debug:
        print('DEBUG Working on two-slice subvolume')
        bssfp = bssfp[:, :, 48:50, :]

    print('Detected %d mag/phase pairs' % int(nt/2))

    # Extract mag and phase volumes
    sm = bssfp[:, :, :, 0::2]
    sp = bssfp[:, :, :, 1::2]

    # Rescale phase from [-4096,4096] to [-pi,pi]
    # Sign change determined empirically (Siemens only)
    sp_rad = -sp * np.pi / 4096.0

    # Convert from mag/phs to re/im complex form
    print('Converting from mag/phase to real/imag complex format')
    sr = sm * np.cos(sp_rad)
    si = sm * np.sin(sp_rad)

    # Create signal mask from MIP of magnitude over all phase advances
    if use_mask:
        print('Using Ostsu MIP mask for calculation')
        mip = np.max(sm, axis=3)
        mask = mip >= threshold_otsu(mip)
    else:
        print('Reconstructing all voxels')
        mask = np.ones_like(sm[:,:,:,0], dtype=np.bool)

    # Run LORE reconstruction over all voxels
    s0, T1, T2, theta = lore(sr, si, mask, tr, te, alpha)

    # Output filenames
    s0_fname = bssfp_fname.replace('.nii.gz', '_s0.nii.gz')
    T1_fname = s0_fname.replace('s0', 'T1')
    T2_fname = s0_fname.replace('s0', 'T2')
    theta_fname = s0_fname.replace('s0', 'theta')

    # Write 3D LORE images
    print('Saving reconstructed images')

    aff_mat = bssfp_nii.get_affine()

    # s0_nii = nib.Nifti1Image(s0, aff_mat)
    # s0_nii.to_filename(s0_fname)
    nib.Nifti1Image(s0, aff_mat).to_filename(s0_fname)

    T1_nii = nib.Nifti1Image(T1, aff_mat)
    T1_nii.to_filename(T1_fname)

    T2_nii = nib.Nifti1Image(T2, aff_mat)
    T2_nii.to_filename(T2_fname)

    theta_nii = nib.Nifti1Image(theta, aff_mat)
    theta_nii.to_filename(theta_fname)

    # Clean exit
    sys.exit(0)


def lore(sr, si, mask, tr, te, alpha):
    """
    Perform LORE reconstruction of complex, multiphase bSSFP data
    :param sr: real part of signal (4D image)
    :param si: imaginary part of signal (4D image)
    :param mask: calculation mask (3D image)
    :param tr: sequence repetition time in ms
    :param te: sequence echo time in ms
    :param alpha: sequence flip angle in degrees
    :return: s0: reconstructed T2w signal without off-resonance effects
    """

    use_mp = True

    # Save 4D shape
    nx, ny, nz, n = sr.shape

    # Setup phase advance vectors
    dth = np.arange(0, n) * (2.0 * np.pi / float(n))
    cos_n = np.cos(dth)
    sin_n = np.sin(dth)

    print('Setting up data for LORE')

    # Flatten spatial dimensions
    mask = mask.reshape(-1)
    sr = sr.reshape(-1, n)
    si = si.reshape(-1, n)

    sr_mask = sr[mask,:]
    si_mask = si[mask,:]

    # Get indices of True elements in mask
    mask_inds = np.where(mask)[0]
    n_vox = len(mask_inds)

    # Create data iterable for starmap LORE processing
    # Should be a list of tuples, one for each voxel in the mask
    # Each tuple should contain the ordered arguments for lore_core()
    lore_data = [(sr[i,:], si[i,:], cos_n, sin_n, tr, te, alpha) for i in mask_inds]

    if use_mp:

        # Use all but two CPUs
        n_cpu = mp.cpu_count() - 2

        print('Running multiprocess LORE estimation on %d voxels (%d CPUs)' % (n_vox, n_cpu))

        # Run LORE processing over multiple processors
        with mp.Pool(n_cpu) as pool:
            res = pool.starmap(lore_core, lore_data)

        # Convert to numpy array and remove any singlet dimensions
        res = np.squeeze(np.array(res))

    else:

        print('Running LORE estimation on %d voxels' % n_vox)

        res = np.zeros([n_vox, 4])

        for vox in range(0,n_vox):
            sr_n, si_n, cos_n, sin_n, tr, te, alpha = lore_data[vox]
            res[vox,:] = lore_core(sr_n, si_n, cos_n, sin_n, tr, te, alpha)

    # Allocate final results
    s0 = np.zeros_like(mask, dtype=float)
    T1 = np.zeros_like(mask, dtype=float)
    T2 = np.zeros_like(mask, dtype=float)
    theta = np.zeros_like(mask, dtype=float)

    # Unpack results
    s0[mask] = res[:,0]
    T1[mask] = res[:,1]
    T2[mask] = res[:,2]
    theta[mask] = res[:,3]

    # Restore 3D shape
    s0 = s0.reshape(nx, ny, nz)
    T1 = T1.reshape(nx, ny, nz)
    T2 = T2.reshape(nx, ny, nz)
    theta = theta.reshape(nx, ny, nz)

    return s0, T1, T2, theta


def lore_core(sr_n, si_n, cos_n, sin_n, tr, te, alpha):
    """
    Core function for multiprocessor LORE estimation of bSSFP signal parameters

    :param sr_n: real signal for one voxel, all phase advances (Nx1)
    :param si_n: imaginary signal for one voxel, all advances (Nx1)
    :param cos_n: cosines of phase advances (Nx1)
    :param sin_n: sines of phase advances (Nx1)
    :param tr: sequence repetition time in ms
    :param te: sequence echo time in ms
    :param alpha: sequence flip angle in degrees
    :return: eta, beta, zeta: estimated LORE parameters for one voxel
    """

    # Number of phase advances
    N = len(sr_n)

    # From Bjork et al Magn Reson Med 2014
    #
    # For N bSSFP acquisitions with phase advances dtheta_n:
    #
    # y_n = [sr_n si_n]
    #
    # A_n = [1 0 -cos_n -sin_n sr_n*cos_n -sr_n*sin_n ]
    #       [0 1  sin_n -cos_n si_n*cos_n -si_n*sin_n ]
    #
    # and, stacking y_n and A_n,
    #
    # y = [y_1 y_2 .. y_n]T (2N x 1 matrix)
    #
    # A = [A_1T A_2T .. A_NT]T (2N x 6 matrix)
    #
    # Finally, we solve for LORE parameters (x)
    #
    # Since, y = Ax
    #
    # x = (AT A)-1.AT.y

    # Init y and A
    y = np.zeros([2*N, 1])
    A = np.zeros([2*N, 6])

    for i in range(0, N):

        # Extract key values
        s, c = sin_n[i], cos_n[i]
        sr, si = sr_n[i], si_n[i]

        # Construct y_n (2x1) and A_n (2x6)
        y_n = np.array([[sr], [si]])
        A_n = np.array([[1, 0, -c, -s, sr * c, -sr * s],
                        [0, 1,  s, -c, si * c, -si * s]])

        # Insert y_n into y and A_n into A
        # Skip double transpose and insert directly as row pairs
        y[(2*i):(2*i+2), :] = y_n
        A[(2*i):(2*i+2), :] = A_n

    # Least-squares solution of y = Ax for x
    x, res, rank, sing = np.linalg.lstsq(A, y)

    # Unpack LORE parameters from x
    eta_r, eta_i, beta_r, beta_i, zeta_r, zeta_i = x

    # Complex valued LORE parameters
    eta = eta_r + 1j * eta_i
    beta = beta_r + 1j * beta_i
    zeta = zeta_r + 1j * zeta_i

    # Estimate signal parameters from LORE parameters

    # Beta/eta ratio - handle divide-by-zero
    if np.abs(eta) < np.finfo(float).eps:
        b_z = 0.0
    else:
        b_z = beta/eta

    # Relaxation parameters
    a = np.abs(b_z, dtype=np.float)
    b = np.abs(zeta, dtype=np.float)

    ca = np.cos(alpha * np.pi / 180.0)

    # TODO: Frequent div-by-zero warnings from this. Possible error in parameter estimation code above
    # Relaxation time estimates
    E1 = (a * (1 + ca - a * b * ca) - b) / (a * (1 + ca - a * b) - b * ca)
    E2 = a

    if E1 > 0.0:
        T1 = -tr / np.log(E1)
    else:
        T1 = 0.0

    if E2 > 0.0:
        T2 = -tr / np.log(E2)
    else:
        T2 = 0.0

    # Phase accumulation per TR due to off-resonance frequency
    theta = -np.angle(b_z)

    # s0 = eta * np.exp(-1j * theta * te_tr)
    s0 = np.abs(eta)

    return s0, T1, T2, theta


# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    main()