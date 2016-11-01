#!/usr/bin/env python3
"""
Inverse FFT a series of mag/phase image pairs in a 4D Nifti.
Output mag/phase k-space images in same order as input data.

Usage
----
swi_kspace.py -i <4D mag/phase volume pairs> -o <4D mag/phase k-space>
swi_kspace.py -h

Authors
----
Mike Tyszka, Caltech Brain Imaging Center

Dates
----
2016-11-01 JMT From scratch

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
from numpy.fft import ifftn, fftshift, ifftshift


def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Complex 3D IFFT of series of mag/phase volumes')
    parser.add_argument('-i', '--input', required=True, help='Input 4D series of 3D mag/phase volumes')

    # Parse command line arguments
    args = parser.parse_args()

    in_fname = args.input

    # Construct output filename from input filename
    if '.nii.gz' in in_fname:
        out_fname = in_fname.replace('.nii.gz', '_kspace.nii.gz')
    elif 'nii' in in_fname:
        out_fname = in_fname.replace('.nii', '_kspace.nii.gz')

    # Load series of 3D mag/phase volumes from 4D Nifti
    print('Loading mag/phase volumes from %s' % in_fname)
    in_nii = nib.load(in_fname)
    s_r = in_nii.get_data()

    # Grab input image dimensions
    # 4th dimension contains nv re/im pairs (ie nt = 2*nv)
    nx, ny, nz, nt = in_nii.header.get_data_shape()
    nv = int(nt/2)

    print('Detected %d mag/phase volumes' % nv)

    # Extract mag and phase volumes
    s_m = s_r[:,:,:,0:nt:2]
    s_p = s_r[:,:,:,1:nt:2]

    # Rescale phase from [-4096,4096] to [-pi,pi]
    s_p_rad = s_p * np.pi / 4096.0

    # Convert from mag/phs to re/im complex form
    s_c = s_m * (np.cos(s_p_rad) + 1.j * np.sin(s_p_rad))

    # Inverse 3D FFT (axes 0,1,2)
    print('Performing inverse 3D FFTs')
    aa = [0,1,2]
    k_c = ifftshift(ifftn(fftshift(s_c, axes=aa), axes=aa), axes=aa)

    # Convert from complex to mag/phase form
    k_m, k_p = np.abs(k_c), np.angle(k_c)

    # Fill 4D mag/phase array
    k_r = np.zeros_like(s_r, dtype=np.float32)
    k_r[:,:,:,0:nt:2] = k_m
    k_r[:,:,:,1:nt:2] = k_p

    # Write mag/phase k-spaces
    print('Saving 3D mag/phase k-spaces to %s' % out_fname)
    swi_nii = nib.Nifti1Image(k_r, in_nii.get_affine())
    swi_nii.to_filename(out_fname)

    # Clean exit
    sys.exit(0)


# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    main()
