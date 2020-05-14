#!/usr/bin/env python3
'''
created on Thu May 15 2020
author: Christian Olaf Haeusler
'''
from glob import glob
import argparse
import os
import subprocess

TNT_DIR = 'inputs/studyforrest-data-templatetransforms'

def parse_arguments():
    '''
    '''
    parser = argparse.ArgumentParser(
        description="converts probabilistic brain lobe masks taken from \
        Jülich Histological Atlas (part of FSL) and converts them into \
        group space")

    parser.add_argument('-i',
                        default='rois-and-masks/mni_prob_*.nii.gz',
                        help='pattern of Jülich Histological Atlas lobe files')

    args = parser.parse_args()

    inPattern = args.i

    return inPattern


def find_files(pattern):
    '''
    '''
    foundFiles = glob(pattern)

    return sorted(foundFiles)


def mni_masks_2_grpbold2Tp(mniMask, outFpath, xfmRef, xfmMat):
    '''transform MNI brain lobe mask to group BOLD 3T phase2 space
    by calling FSL's flirt -applyxfm
    '''

    xfmInterp = 'trilinear'

    if not os.path.exists(outFpath):
        subprocess.call(
            ['flirt',
             '-applyxfm',
             '-in', mniMask,
             '-out', outFpath,
             '-ref', xfmRef,
             '-init', xfmMat,
             '-interp', xfmInterp,
             '-paddingsize', '0.0',
             ]
        )

    return None


if __name__ == "__main__":
    # get the command line inputs
    inPattern = parse_arguments()

    lobeFpathes = find_files(inPattern)

    # get the transformation matrix and reference image
    xfmMat = os.path.join(TNT_DIR, 'templates/grpbold3Tp2/xfm/mni2tmpl_12dof.mat')
    xfmRef = os.path.join(TNT_DIR, 'templates/grpbold3Tp2/brain.nii.gz')

    if not os.path.exists(xfmRef):
        subprocess.call(['datalad', 'get', xfmRef])

    if not os.path.exists(xfmMat):
        subprocess.call(['datalad', 'get', xfmMat])

    # transformation the brain lobe masks from MNI space to grpbold3Tp2 space
    # by calling FSL's flirt -applyxfm
    for lobeFpath in lobeFpathes:
        outFpath = lobeFpath.replace('mni_prob', 'grpbold3Tp2_mni_prob')
        mni_masks_2_grpbold2Tp(lobeFpath, outFpath, xfmRef, xfmMat)
