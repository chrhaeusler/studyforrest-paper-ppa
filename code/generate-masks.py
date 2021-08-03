#!/usr/bin/env python3
'''
author: Christian Olaf Häusler
created on Thursday April 18 2019

script does for group:
1. creates bilateral, probabilistic & binary group masks

script does for individual subjects
1. create bilateral, probabilistic & binarized PPA masks from group masks
2. transforms probabilistic MNI brain lobe masks to subjects' spaces
3. merge the temporal and occipital MNI masks in subject space
4. creates FOV masks for individual brains from 4d 7 tesla time series
5. join & transforms individual PPA masks into group space
...and download all data via 'datalad get' in case data are not on disk
'''

from glob import glob
import argparse
import subprocess
import nibabel as nib
import numpy as np
import os
import re


def parse_arguments():
    '''
    '''
    parser = argparse.ArgumentParser(
        description='creates lots of masks'
    )

    parser.add_argument('-out',
                        required=False,
                        default='rois-and-masks',
                        help='the output directory (e.g. "rois-and-masks")')

    args = parser.parse_args()

    out_dir = args.out

    return out_dir


def find_files(pattern):
    '''
    '''
    found_files = glob(pattern)
    found_files = sort_nicely(found_files)

    return found_files


def sort_nicely(l):
    '''Sorts a given list in the way that humans expect
    '''
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    l.sort(key=alphanum_key)

    return l


def create_grp_bilat_mask(grp_masks, grp_out):
    '''merges unilateral ROI overlaps to bilateral probabilistic & binary image
    '''
    # create the bilateral probabilistic image
    if not os.path.exists(grp_out):
        mask_img = nib.load(grp_masks[0])
        mask_data = np.array(mask_img.dataobj)

        # if there is more than one mask, load all of them and take their sum
        if len(grp_masks) > 1:
            for mask in grp_masks[1:]:
                mask_data += np.array(nib.load(mask).dataobj)

        # create the probabilistic image to be saved
        bilat_mask_img = nib.Nifti1Image(mask_data,
                                         mask_img.affine,
                                         header=mask_img.header)

        # save the bilateral probabilistic image
        nib.save(bilat_mask_img, grp_out)

    # binarize the bilateral probabilistic image
    grp_out = grp_out.replace('prob', 'binary')

    if not os.path.exists(grp_out):
        # make the data binary by setting all non-zeros to 1
        mask_data[mask_data > 0] = 1

        # create the binary image to be saved
        bilat_mask_img = nib.Nifti1Image(mask_data,
                                         mask_img.affine,
                                         header=mask_img.header)
        # save the image
        nib.save(bilat_mask_img, grp_out)

        print('wrote', grp_out)

    return None


def grp_ppa_to_ind_space(in_file, output, ref, warp):
    '''
    '''
    # only process if output file does not exists
    if os.path.exists(output):
        print(output, 'already exists. skipping...')
    else:
        # download necessaty input
        if not os.path.exists(ref):
            subprocess.call(['datalad', 'get', ref])

        if not os.path.exists(warp):
            subprocess.call(['datalad', 'get', warp])

        subprocess.call(
            ['applywarp',
             '-i', in_file,
             '-o', output,
             '-r', ref,
             '-w', warp,
             # '--premat=premat'
             ])

    return None


def mni_masks_2_bold3Tp2(juelMNI, outFpath, indRef, indWarp, premat):
    '''
    '''
    if not os.path.exists(indRef):
        subprocess.call(['datalad', 'get', indRef])

    if not os.path.exists(indWarp):
        subprocess.call(['datalad', 'get', indWarp])

    if not os.path.exists(premat):
        subprocess.call(['datalad', 'get', premat])

    subprocess.call(
        ['applywarp',
         '-i', juelMNI,
         '-o', outFpath,
         '-r', indRef,
         '-w', indWarp,
         '--premat=' + premat
         ]
    )

    return None


def combine_mni_masks(indBrain, occFpath, tempFpath, outFpath):
    '''
    '''
    # load the three masks
    mask_brain = nib.load(indBrain)
    mask_occip = nib.load(occFpath)
    mask_tempo = nib.load(tempFpath)

    # get array with actual data
    brain_array = mask_brain.get_fdata()

    occip_header = mask_occip.header
    occip_affine = mask_occip.affine
    occip_array = mask_occip.get_fdata()

    # tempo_header = mask_tempo.header
    # tempo_affine = mask_tempo.affine
    tempo_array = mask_tempo.get_fdata()

    # unify the individuum's temporal and occipital mask
    occip_tempo_mask = occip_array + tempo_array
    # multiply the probabilistic map with the actual brain mask
    mask_outp = occip_tempo_mask * brain_array

    # save that shit to file
    try:
        nib.save(nib.Nifti1Image(
            mask_outp,
            occip_affine,
            occip_header),
            outFpath)
    except PermissionError as e:
        print(e)

    return None


def create_audio_mask(aoPattern, subj, ao_mask_fpath):
    '''
    '''
    audioFpath = aoPattern.replace('###SUB###', subj)

    if not os.path.exists(audioFpath):
        subprocess.call(['datalad', 'get', audioFpath])

    audio_4d_img = nib.load(audioFpath)
    audio_4d = audio_4d_img.get_fdata()
    # slice time series for one volume
    # take the 20 volume due to possible artifacts at the experiment's
    # beginning
    audio_mask = np.copy(audio_4d[...,20])
    # create a mask of 1s and 0s
    ### der eigentliche Wert für außerhalb des fov ist 8
    ### value 16 klappt für alle außer VP 06
    if subj != 'sub-06':
        audio_mask[audio_mask <= 16] = 0
        audio_mask[audio_mask > 16] = 1

    else:
        audio_mask[audio_mask <= 400] = 0
        audio_mask[audio_mask > 400] = 1

    try:
        nib.save(nib.Nifti1Image(
            audio_mask,
            audio_4d_img.affine,
            audio_4d_img.header),
            ao_mask_fpath)
    except PermissionError as e:
        print(e)

    return None


def create_ind_bilat_mask(indUniMasksFpath, indBilatMaskFpath):
    '''
    '''
    oneUniImg = nib.load(indUniMasksFpath[0])
    oneUniData = np.array(oneUniImg.dataobj)

    # if there is more than one mask, load all of them and take their sum
    if len(indUniMasksFpath) > 1:
        for mask in indUniMasksFpath[1:]:
            oneUniData += np.array(nib.load(mask).dataobj)

    biData = oneUniData

    # prepare to save
    bilatMask = nib.Nifti1Image(biData,
                                oneUniImg.affine,
                                header=oneUniImg.header
                                )

    try:
        nib.save(bilatMask, indBilatMaskFpath)
    except PermissionError as e:
        print(e)

    return None


def warp_subj_to_mni(indBiMaskFpath, indBiInMNI, indWarp, xfmRef):
    '''
    '''
    if not os.path.exists(indWarp):
        subprocess.call(['datalad', 'get', indWarp])

    if not os.path.exists(xfmRef):
        subprocess.call(['datalad', 'get', xfmRef])

    # call FSL's applywarp
    subprocess.call(
        ['applywarp',
         '-i', indBiMaskFpath,
         '-o', indBiInMNI,
         '-r', xfmRef,
         '-w', indWarp,
         # '--premat=premat'
         ]
    )


# main program #
if __name__ == "__main__":
    # some hardcoded sources
    # input directory of Sengupta et al.
    SENGUPTA = 'inputs/studyforrest-data-visualrois'

    # input directory of the current paper's analysis
    PPA_DIR = 'inputs/studyforrest_ppa'
    # an exemplary 4D image of the audio-descriptions 4D data
    AUDIO_4D_EXAMPLE = os.path.join(
        PPA_DIR,
        'inputs/studyforrest-data-aligned/'\
        '###SUB###/in_bold3Tp2/###SUB###_task-aomovie_run-1_bold.nii.gz'
    )

    # input directory of templates & transforms
    TNT_DIR = 'inputs/studyforrest-data-templatetransforms'
    # group BOLD image (reference image)
    xfmRef = os.path.join(TNT_DIR,
                          'templates/grpbold3Tp2/',
                          'brain.nii.gz')

    xfmMat = os.path.join(TNT_DIR,
                          'templates/grpbold3Tp2/xfm/',
                          'mni2tmpl_12dof.mat')

    ROIS = 'rois-and-masks'

    # the output directory
    out_dir = parse_arguments()

    # 1. create bilateral, probabilistic & binary group masks
    # input: Juelich Histological Atlas
    juelMNIoccip = os.path.join(ROIS, 'mni_prob_occipital_lobe.nii.gz')
    juelMNItempo = os.path.join(ROIS, 'mni_prob_temporal_lobe.nii.gz')

    # input: ROIs from Sengupta et al. (2016)
    uniGrpMaskPattern = os.path.join(ROIS, '?PPA_overlap.nii.gz')
    grpMasksFpathes = find_files(uniGrpMaskPattern)

    # output
    grpMaskFpath = 'rois-and-masks/bilat_PPA_prob.nii.gz'

    # do the conversion
    create_grp_bilat_mask(grpMasksFpathes, grpMaskFpath)

    # process individuals
    print('\nprocessing individuals')
    # find all individual z-maps files across individual subjects
    AO_ZMAP_PATTERN = os.path.join(
        PPA_DIR,
        'sub-??/2nd-lvl_audio-ppa-ind.gfeat/cope*.feat/stats/zstat1.nii.gz')

    AV_ZMAP_PATTERN = os.path.join(
        PPA_DIR,
        'sub-??/2nd-lvl_movie-ppa-ind.gfeat/cope*.feat/stats/zstat1.nii.gz')

    aoFpathes = find_files(AO_ZMAP_PATTERN)
    avFpathes = find_files(AV_ZMAP_PATTERN)
    subjsFpathes = aoFpathes + avFpathes

    # get a list of subjects (as strings)
    subjs = [re.search(r'sub-..', string).group() for string in aoFpathes]
    subjs = sorted(list(set(subjs)))

    for subj in subjs:
        print('\nProcessing', subj)
        # 1. create bilateral, probabilistic & binarized PPA masks
        # from group masks (in group space)

        # create subject-specific subfolder first
        indBrain = os.path.join(TNT_DIR, subj, 'bold3Tp2/brain.nii.gz')
        indGrpPPA = os.path.join(ROIS, '###SUB###/grp_PPA_bin.nii.gz')
        indGrpPPA = indGrpPPA.replace('###SUB###', subj)

        indRef = os.path.join(TNT_DIR, '###SUB###/bold3Tp2/brain.nii.gz')
        indRef = indRef.replace('###SUB###', subj)
        indWarp = os.path.join(TNT_DIR, '###SUB###/bold3Tp2/',
                               'in_grpbold3Tp2/tmpl2subj_warp.nii.gz')
        indWarp = indWarp.replace('###SUB###', subj)

        os.makedirs(os.path.dirname(indGrpPPA), exist_ok=True)

        print('transform group PPA into subject space')
        grp_ppa_to_ind_space(grpMaskFpath, indGrpPPA, indRef, indWarp)

        # 2. transform MNI brain lobe masks to individual subjects' spaces
        print('\nconvert probabilistic mask from MNI to subject space')

        indRef = os.path.join(TNT_DIR, subj, 'bold3Tp2/brain.nii.gz')
        templ2subjWarp = os.path.join(TNT_DIR, subj,
                                      'bold3Tp2/in_grpbold3Tp2/'
                                      'tmpl2subj_warp.nii.gz')

        # convert the occipital mask from MNI to individual bold3Tp2
        indOccip = os.path.basename(juelMNIoccip)
        indOccip = os.path.join(ROIS, subj, indOccip)
        mni_masks_2_bold3Tp2(juelMNIoccip, indOccip,
                             indRef, templ2subjWarp, xfmMat)

        # convert the temporal mask from MNI to individual bold3Tp2
        indTempo = os.path.basename(juelMNItempo)
        indTempo = os.path.join(ROIS, subj, indTempo)
        mni_masks_2_bold3Tp2(juelMNItempo, indTempo, indRef,
                             templ2subjWarp, xfmMat)

        # 3. merge the temporal and occipital masks in subject space
        # create name of output file
        indOccTemp = os.path.dirname(indTempo)
        indOccTemp = os.path.join(indOccTemp, 'mni_prob_occip_tempo.nii.gz')
        # combine the two masks
        combine_mni_masks(indBrain, indOccip, indTempo, indOccTemp)

        # 4. create FOV mask from 4d 7 tesla time series
        print('\ncreate FOV mask (Hanke et al., 2014)')
        ao_mask_fpath = os.path.join(ROIS, subj, 'ao_fov_mask.nii.gz')
        create_audio_mask(AUDIO_4D_EXAMPLE, subj, ao_mask_fpath)

        # filter globbed pathes/files for current subject
        subj_fpathes = [zmap for zmap in subjsFpathes if subj in zmap]

        # ROI masks
        # 5. joins & transforms individual PPA masks into group space
        # (ROI masks are taken from Sengupta et al. (2016)
        print('\nconvert individual ROI masks '
              '(Sengupata et al., 2016) to group space')

        # find left and right hemispheric masks
        indMaskPattern = os.path.join(SENGUPTA,
                                      subj,
                                      'rois/?PPA_?_mask.nii.gz')
        # do the globbing
        indUniMasksFpath = find_files(indMaskPattern)

        # create the output name
        indBiMaskFpath = os.path.join(ROIS,
                                      subj,
                                      'PPA_%s.nii.gz' % len(indUniMasksFpath)
                                      )

        # if mask(s for PPA) exists, process it/them
        # join individual unilateral ROI clusters
        # to bilateral mask & transform into subject space
        create_ind_bilat_mask(indUniMasksFpath, indBiMaskFpath)

        # warp bilateral ROI clusters to MNI group space
        indBiMNIfPath = indBiMaskFpath.replace('.nii.gz', '_mni.nii.gz')
        # name if current's subjects warp volume
        subj2templWarp = os.path.join(TNT_DIR,
                                      subj,
                                      'bold3Tp2/in_grpbold3Tp2/'
                                      'subj2tmpl_warp.nii.gz'
                                      )

        warp_subj_to_mni(indBiMaskFpath, indBiMNIfPath,
                            subj2templWarp, xfmRef)

        # binarize the individual bilateral ROI clusters (in MNI space)
        indBiMNIbinFpath = indBiMNIfPath.replace('mni', 'mni_bin')
        # call fslmath
        subprocess.call(['fslmaths',
                            indBiMNIfPath,
                            '-bin',
                            indBiMNIbinFpath]
                        )

        # individuals to t1w
        out_path = os.path.join(out_dir, subj, 'in_t1w')
        os.makedirs(out_path, exist_ok=True)

        highres_ref = os.path.join(TNT_DIR, subj, 't1w/brain.nii.gz')
        in_matrix = os.path.join(TNT_DIR, subj, 'bold3Tp2/in_t1w/xfm_6dof.mat')

        for to_get in [highres_ref, in_matrix]:
            if not os.path.exists(to_get):
                subprocess.call(['datalad', 'get', to_get])

        # individual PPA from bold3Tp2 to t1w
        print('\nconvert individual PPA mask from bold3Tp2 to t1w')
        subprocess.call(
            ['flirt',
             '-in', indBiMaskFpath,
             '-out', os.path.join(out_path, os.path.basename(indBiMaskFpath)),
             '-ref', highres_ref,
             '-applyxfm',
             '-init', in_matrix,
             '-interp', 'nearestneighbour'
             # '--premat=premat'
             ]
        )

        # audio-description's contrast 1 (z=2.3) from bold3Tp2 to t1w
        inFpath = AO_ZMAP_PATTERN.replace('sub-??', subj).replace('cope*', 'cope1')
        inFpath = inFpath.replace('stats/zstat1.nii.gz', 'thresh_zstat1.nii.gz')

        if not os.path.exists(inFpath):
                subprocess.call(['datalad', 'get', inFpath])

        print('\nconvert audio-description\'s primary COPE from bold3Tp2 to t1w')
        subprocess.call(
            ['flirt',
             '-in', inFpath,
             '-out', os.path.join(out_path, 'ao-cope1-ind.nii.gz'),
             '-ref', highres_ref,
             '-applyxfm',
             '-init', in_matrix,
             '-interp', 'nearestneighbour'
             # '--premat=premat'
             ]
        )

        # audio-description's contrast 1 (z=3.4) from grpbold3Tp2 to t1w
        print('\nconvert audio-description\'s primary COPE from grpbold3Tp2 to t1w')

        # define inputs
        inFpath = inFpath.replace('-ind', '-grp')
        in_matrix = os.path.join(TNT_DIR, subj, 't1w/in_mni152/tmpl2subj.mat')
        templ2subjWarp = os.path.join(TNT_DIR, subj, 'bold3Tp2/in_grpbold3Tp2/tmpl2subj_warp.nii.gz')
        postmat = os.path.join(TNT_DIR, subj, 'bold3Tp2/in_t1w/xfm_6dof.mat')

        # download the necessary files
        for to_get in [inFpath, in_matrix, templ2subjWarp, postmat]:
            if not os.path.exists(to_get):
                subprocess.call(['datalad', 'get', to_get])

        subprocess.call(
            ['applywarp',
             '-i', inFpath,
             '-o', os.path.join(out_path, 'ao-cope1-grp.nii.gz'),
             '-r', highres_ref,
             '-w', templ2subjWarp,
             f'--postmat={postmat}',
             '--interp=nn'  # nearest neighbour
             ])


        # movie's contrast 1 (z=2.3) from bold3Tp2 to t1w
        inFpath = AV_ZMAP_PATTERN.replace('sub-??', subj).replace('cope*', 'cope1')
        inFpath = inFpath.replace('stats/zstat1.nii.gz', 'thresh_zstat1.nii.gz')

        if not os.path.exists(inFpath):
                subprocess.call(['datalad', 'get', inFpath])

        print('\nconvert movie\'s primary COPE from bold3Tp2 to t1w')
        subprocess.call(
            ['flirt',
             '-in', inFpath,
             '-out', os.path.join(out_path, 'av-cope1-ind.nii.gz'),
             '-ref', highres_ref,
             '-applyxfm',
             '-init', in_matrix,
             '-interp', 'nearestneighbour'
             # '--premat=premat'
             ]
        )

        # movie's contrast 1 (z=3.4) from grpbold3Tp2 to t1w
        print('\nconvert movie\'s primary COPE from grpbold3Tp2 to t1w')

        # define inputs
        inFpath = inFpath.replace('-ind', '-grp')
        in_matrix = os.path.join(TNT_DIR, subj, 't1w/in_mni152/tmpl2subj.mat')
        templ2subjWarp = os.path.join(TNT_DIR, subj, 'bold3Tp2/in_grpbold3Tp2/tmpl2subj_warp.nii.gz')
        postmat = os.path.join(TNT_DIR, subj, 'bold3Tp2/in_t1w/xfm_6dof.mat')

        # download the necessary files
        for to_get in [inFpath, in_matrix, templ2subjWarp, postmat]:
            if not os.path.exists(to_get):
                subprocess.call(['datalad', 'get', to_get])

        subprocess.call(
            ['applywarp',
             '-i', inFpath,
             '-o', os.path.join(out_path, 'av-cope1-grp.nii.gz'),
             '-r', highres_ref,
             '-w', templ2subjWarp,
             f'--postmat={postmat}',
             '--interp=nn'  # nearest neighbour
             ])
