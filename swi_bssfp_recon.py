#!/usr/bin/env python3
"""
Convert a series of Siemens mag/phase bSSFP images to real/imag form
and combine into a single complex volume. Perform low pass spatial filtering
on combined complex volume prior to SWI phase reconstruction

Usage
----
swi_bssfp_recon.py -i <4D bSSFP mag/phase volumes> -o <3D re/im volume>
swi_bssfp_recon.py -h

Authors
----
Mike Tyszka, Caltech Brain Imaging Center

Dates
----
2016-10-29 JMT From scratch

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

__version__ = '0.1.0'

import sys
import argparse
import nibabel as nib
import numpy as np


def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Multi-phase bSSFP SWI reconstruction')
    parser.add_argument('-i', '--input', required=True, help='Input 4D bSSFP volumes in mag/phase format')

    # Parse command line arguments
    args = parser.parse_args()

    in_fname = args.input

    # Construct SWI filename from bSSFP filename
    if '.nii.gz' in in_fname:
        swi_fname = in_fname.replace('.nii.gz', '_swi.nii.gz')
    elif 'nii' in in_fname:
        swi_fname = in_fname.replace('.nii', '_swi.nii.gz')

    # Load 4D mag/phase bSSFP volumes
    print('Loading bSSFP mag/phase volumes from %s' % in_fname)
    in_nii = nib.load(in_fname)
    bssfp = in_nii.get_data()

    # Grab input image dimensions
    # 4th dimension contains nv re/im pairs (ie nt = 2*nv)
    nx, ny, nz, nt = in_nii.header.get_data_shape()
    nv = int(nt/2)

    # Extract mag and phase volumes
    bssfp_mag = bssfp[:,:,:,0]
    bssfp_phi = bssfp[:,:,:,1]

    # Rescale phase from [-4096,4096] to [-pi,pi]
    bssfp_phi_rad = bssfp_phi * np.pi / 4096.0

    # Convert from mag/phs to re/im complex form
    bssfp_c = bssfp_mag * (np.cos(bssfp_phi_rad) + 1.j * np.sin(bssfp_phi_rad))

    # High pass filter complex image - removes low spatial frequency mag and phase components
    bssfp_c_hp = hpf(bssfp_c, sigma=5.0)

    # Take complex voxel-wise median over all volumes
    bssfp_c_hp_med = np.median(bssfp_c_hp, axis=3)

    # SWI reconstruction from median bSFFP complex volume
    swi = np.abs(bssfp_c_hp_med)
    # swi = np.angle(bssfp_c_hp_med, deg=True)
    # swi = np.real(bssfp_c_hp_med)
    # swi = np.imag(bssfp_c_hp_med)

    # Write 3D complex-valued SWI
    print('Saving reconstructed SWI to %s' % swi_fname)
    swi_nii = nib.Nifti1Image(swi, in_nii.get_affine())
    swi_nii.to_filename(swi_fname)

    # Clean exit
    sys.exit(0)


def hpf(Sr_c, sigma=5.0):
    """
    High pass filter in complex Fourier domain

    :param img_c: original complex image
    :param sigma: Gaussian filter sigma (voxels)
    :return img_c_hpf: high pass filtered complex image
    """

    # Inverse FFT (to time domain k-space)
    Sk_c = np.ifftn(Sr_c)

    # High pass filter
    Sk_c_hpf = Sk_c * 1.0

    # Forward FFT (to frequence domain)
    Sr_c_hpf = np.fft(Sk_c_hpf)

    return Sr_c_hpf


# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    main()
