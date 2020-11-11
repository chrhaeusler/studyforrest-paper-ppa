#!/usr/bin/env python2
# -*- coding: utf-8 -*-
'''
author: Christian Olaf Häusler
created on Monday June 29 2020

start ipython2.7 via
'python2 -m IPython'

'''

from __future__ import print_function
from glob import glob
import argparse
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
import re
import subprocess
from mvpa2.datasets.mri import fmri_dataset
from scipy.stats import gaussian_kde

# check some stuff first
print(os.getcwd())

# bilateral PPA group mask
# PPA_GRP_MASK = 'rois-and-masks/bilat_PPA_binary.nii.gz'

# individual FOV mask of the audio drama fMRI data
AO_FOV_PATTERN = 'rois-and-masks/sub-*/ao_fov_mask.nii.gz'

# individual mask comprising occipital and temporal cortex
OCCTEMP_MASK_PATTERN = 'rois-and-masks/sub-*/mni_prob_occip_tempo.nii.gz'

# individual, bilateral PPA group mask but in subjects space
PPA_GRP_SUBJ_PATTERN = 'rois-and-masks/sub-*/grp_PPA_bin.nii.gz'

# binary mask(s) of individual visual localizer (in subject space)
PPA_MASK_PATTERN = 'inputs/studyforrest-data-visualrois/'\
    'sub-*/rois/?PPA_?_mask.nii.gz'
# unthresholded zmap of individual visual localizer (in subject space)
VLOC_ZMAP_PATTERN = 'inputs/studyforrest-data-visualrois/'\
    'sub-*/2ndlvl.gfeat/cope*.feat/stats/zstat1.nii.gz'

# contrast used by Sengupta et al. (2016) to create the PPA mask
VLOC_VPN_COPES = {
    'sub-01': 'cope8',
    'sub-02': 'cope3',
    'sub-03': 'cope3',
    'sub-04': 'cope3',
    'sub-05': 'cope3',
    'sub-06': 'cope3',
    'sub-09': 'cope3',
    'sub-14': 'cope3',
    'sub-15': 'cope3',
    'sub-16': 'cope3',
    'sub-17': 'cope3',
    'sub-18': 'cope8',
    'sub-19': 'cope3',
    'sub-20': 'cope3'
}

# individual 2nd level results (primary cope in subject space)
AUDIO_ZMAP_PATTERN = 'inputs/studyforrest_ppa/sub-*/'\
    '2nd-lvl_audio-ppa-ind.gfeat/cope1.feat/stats/zstat1.nii.gz'
# MOVIE_ZMAP_PATTERN = 'inputs/studyforrest_ppa/sub-*/'\
#     '2nd-lvl_movie-ppa-ind.gfeat/cope1.feat/stats/zstat1.nii.gz'


def parse_arguments():
    '''
    '''
    parser = argparse.ArgumentParser(
        description='Creates mosaic of individual Bland-Altman-Plots'
    )

    parser.add_argument('-o',
                        default='paper/figures/',
                        help='output directory')

    args = parser.parse_args()

    outDir = args.o

    return outDir


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


def load_subj_ppa_mask(subj, combined_mask):
    '''
    '''
    # filter PPA masks of all subjects for current subject only
    ppa_fpathes = find_files(PPA_MASK_PATTERN.replace('###SUB###', subj))

    # combine current subject's left & right PPA mask into one mask
    ppa_mask = fmri_dataset(ppa_fpathes,
                            mask=combined_mask).samples.sum(axis=0)

    return ppa_mask


def compute_means(data1, data2, log='n'):
    '''
    '''
    if len(data1) != len(data2):
        raise ValueError('data1 does not have the same length as data2.')

    data1 = np.asarray(data1)
    data2 = np.asarray(data2)

    if log == 'n':
        means = np.mean([data1, data2], axis=0)
    elif log == 'y':
        # what ever computation
        pass

    return means


def compute_diffs(data1, data2, log='n'):
    '''
    '''
    if len(data1) != len(data2):
        raise ValueError('data1 does not have the same length as data2.')

    data1 = np.asarray(data1)
    data2 = np.asarray(data2)

    if log == 'n':
        diffs = data1 - data2  # Difference between data1 and data2
    elif log == 'y':
        # what ever computation
        pass

    return diffs


def process_df(subj, zmaps_df, out_path):
    '''
    '''
    # get the contrasts' names from the column names
    zmap1 = zmaps_df.iloc[:, 1]
    zmap2 = zmaps_df.iloc[:, 0]

    # mask all voxels not contained in the PPA group mask
    ppa_grp_masked1 = zmap1.as_matrix()[zmaps_df['ppa_grp'].as_matrix() > 0]
    ppa_grp_masked2 = zmap2.as_matrix()[zmaps_df['ppa_grp'].as_matrix() > 0]

    # mask all voxels not contained in the individual PPA mask
    ppa_ind_masked1 = zmap1.as_matrix()[zmaps_df['ppa_ind'].as_matrix() > 0]
    ppa_ind_masked2 = zmap2.as_matrix()[zmaps_df['ppa_ind'].as_matrix() > 0]

    ao_ind_masked1 = zmap1.as_matrix()[zmaps_df['ao_ind'].as_matrix() > 0]
    ao_ind_masked2 = zmap2.as_matrix()[zmaps_df['ao_ind'].as_matrix() > 0]

    datasets = [
        [zmap1, zmap2],
        [ppa_grp_masked1, ppa_grp_masked2],
        [ppa_ind_masked1, ppa_ind_masked2],
        [ao_ind_masked1, ao_ind_masked2]
    ]

    means_list = [compute_means(data1, data2) for data1, data2 in datasets]
    diffs_list = [compute_diffs(data1, data2) for data1, data2 in datasets]

    # Set up the axes with gridspec
    fig = plt.figure(figsize=(6, 6))
    grid = plt.GridSpec(6, 6, hspace=0.0, wspace=0.0)

    # add three subplots
    ax_scatter = fig.add_subplot(grid[1:, :-1])
    ax_xhist = fig.add_subplot(grid[0:1, 0:-1],
                               yticklabels=[],
                               sharex=ax_scatter)
    ax_yhist = fig.add_subplot(grid[1:, -1],
                               xticklabels=[],
                               sharey=ax_scatter)

    ax_scatter.text(5.1, 5.8, subj, fontsize=16, fontweight='bold')

    # plot voxel within occipitotemporal cortex
    plot_blandaltman(ax_scatter,
                     means_list[0],
                     diffs_list[0],
                     alpha=0.6,
                     c='darkgrey',
                     s=2)

    # plot voxels within PPA group overlap
    plot_blandaltman(ax_scatter,
                     means_list[1],
                     diffs_list[1],
                     alpha=1,
                     c='royalblue',
                     s=2)

    # plot voxels within individual PPA ROI
    plot_blandaltman(ax_scatter,
                     means_list[2],
                     diffs_list[2],
                     alpha=1,
                     c='r',
                     s=2)

    # plot voxels within (thresholded) individual AO zmap
    plot_blandaltman(ax_scatter,
                     means_list[3],
                     diffs_list[3],
                     alpha=0.5,
                     c='y',
                     s=2)

    plot_histogram(ax_xhist, ax_yhist,
                   means_list[0], diffs_list[0],
                   alpha=1,
                   color='darkgrey')

    plot_histogram(ax_xhist, ax_yhist,
                   means_list[1], diffs_list[1],
                   alpha=1,
                   color='royalblue')

    plot_histogram(ax_xhist, ax_yhist,
                   means_list[2], diffs_list[2],
                   alpha=1,
                   color='r')

    try:
        plot_histogram(ax_xhist, ax_yhist,
                       means_list[3], diffs_list[3],
                       alpha=1,
                       color='y')

    except ValueError:
        print(subj, 'has no significant cluster in primary AO contrast')

    # save that shit
    out_file = ('%s_bland-altman.png' % subj)
    out_fpath = os.path.join(out_path, out_file)

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    plt.savefig(out_fpath,
                bbox_inches='tight',
                dpi=80)
    plt.close()


def plot_blandaltman(ax, means, diffs, *args, **kwargs):
    '''
    '''
    if len(means) != len(diffs):
        raise ValueError('means do not have the same length as diffs.')

    # annotation
    # variable subj is still a global here
    if subj in ['sub-01', 'sub-04', 'sub-09', 'sub-16', 'sub-19']:
        ax.set_ylabel('Difference between 2 measures', fontsize=16)

    if subj in ['sub-19', 'sub-20']:
        ax.set_xlabel('Average of 2 measures', fontsize=16)

    # draw the scattergram
    ax.scatter(means, diffs, *args, **kwargs)

    # set the size of the tick labels
    ax.xaxis.set_tick_params(labelsize=16)
    ax.yaxis.set_tick_params(labelsize=16)

    # limit the range of data shown
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)

    # draw horizontal and vertical line at 0
    ax.axhline(0, color='k', linewidth=0.5, linestyle='--')
    ax.axvline(0, color='k', linewidth=0.5, linestyle='--')

    return None


def plot_histogram(x_hist, y_hist, means, diffs, *args, **kwargs):
    '''
    '''
    # basic preparation of the axes
    x_hist.xaxis.set_tick_params(bottom=True, labelbottom=False)
    y_hist.yaxis.set_tick_params(bottom=True, labelbottom=False)

    # show vertical/horizontal zero line
    x_hist.axvline(0, color='k', linewidth=0.5, linestyle='--')
    y_hist.axhline(0, color='k', linewidth=0.5, linestyle='--')

    # plot histogram -> take KDE plot, s. below
    # x_hist.hist(means, 50, normed=True, histtype='bar',
    #             orientation='vertical', **kwargs)
    # y_hist.hist(diffs, 50, normed=True, histtype='bar',
    #             orientation='horizontal', **kwargs)

    # x_hist.set_yscale('log')
    # y_hist.set_xscale('log')

    # plot KDE
    xvalues = np.arange(-5.1, 5.1, 0.1)
    kde_means = gaussian_kde(means)
    kde_diffs = gaussian_kde(diffs)

    # KDE subplot on the top
    x_hist.plot(xvalues, kde_means(xvalues), **kwargs)
    # x_hist.fill_between(xvalues, kde_means(xvalues), 0, **kwargs)
    x_hist.set_ylim(0.015, 0.8)
    x_hist.set_yticks([0.2, 0.4, 0.6])  # , 1])
    x_hist.set_yticklabels(['.2', '.4', '.6'])  # , '1'])

    x_hist.yaxis.set_tick_params(labelsize=16)

    # KDE subplot on the right
    y_hist.plot(kde_diffs(xvalues), xvalues, **kwargs)
    # y_hist.fill_between(xvalues, kde_diffs(xvalues), 0, **kwargs)
    y_hist.set_xlim(0.015, .8)
    y_hist.set_xticks([0.2, 0.4, 0.6])  # , 1])
    y_hist.set_xticklabels(['.2', '.4', '.6'])  # , '1'])

    y_hist.xaxis.set_tick_params(labelsize=16)

    return None


def create_mosaic(in_pattern, dims, out_fpath, dpi):
    '''
    http://www.imagemagick.org/Usage/montage/
    '''

    # dimensions in columns*rows
    subprocess.call(
        ['montage',
         '-density', str(dpi),
         in_pattern,
         '-geometry', '+1+1',
         '-tile', dims,
         out_fpath])


if __name__ == "__main__":
    # get command line argument
    outDir = parse_arguments()

    # get pathes & filenames of all available zmaps
    audio_fpathes = find_files(AUDIO_ZMAP_PATTERN)
    subjs = [re.search(r'sub-..', string).group() for string in audio_fpathes]
    subjs = sorted(list(set(subjs)))

    for subj in subjs[:]:  # use subject 14 only [7:8]
        print('\nProcessing', subj)
        # load subject-specific FOV used during scannign of audio-description
        aoFOVimg = nib.load(AO_FOV_PATTERN.replace('sub-*', subj))
        aoFOVdata = aoFOVimg.get_data()

        # load subject-specific occipito-temporal mask
        # (probabilistic mask from Jülich Histological Atlas
        occTempMaskFpath = OCCTEMP_MASK_PATTERN.replace('sub-*', subj)
        occTempMaskData = nib.load(occTempMaskFpath).get_data()
        occTempMaskData = occTempMaskData * aoFOVdata

        # initialize a dataframe that will contain
        # unthresholded zmap of primary AO contrast
        # unthresholded zmap of chosen ROI contrast (Sengupta et al. 2016)
        # the group overlap of individual PPA ROIs (Sengupta et al., 2016)
        # the actual individual PPA mask (Sengupta et al., 2016)
        # all images will be masked with the audio-descriptions FoV
        zmaps_df = pd.DataFrame()

        # current subject's primary PPA contrast using descriptive nouns
        aoFpath = AUDIO_ZMAP_PATTERN.replace('sub-*', subj)
        if not os.path.exists(aoFpath):
            subprocess.call(['datalad', 'get', aoFpath])

        aoData = fmri_dataset(aoFpath,
                              mask=occTempMaskData).samples.T
        zmaps_df['aoData'] = np.ndarray.flatten(aoData)

        # current subject's contrast chosen by Sengupta et al. (2016)
        # to create the individual PPA mask
        roiFpath = VLOC_ZMAP_PATTERN.replace('sub-*', subj)
        roiFpath = roiFpath.replace('cope*', VLOC_VPN_COPES[subj])
        roiData = fmri_dataset(roiFpath,
                               mask=occTempMaskData).samples.T
        zmaps_df['roiData'] = np.ndarray.flatten(roiData)

        # the group overlap of individual PPA ROIs
        # but in current subject's space
        grp_ppa_subj = PPA_GRP_SUBJ_PATTERN.replace('sub-*', subj)
        ppaGrpData = fmri_dataset(grp_ppa_subj,
                                  mask=occTempMaskData).samples.T
        zmaps_df['ppa_grp'] = np.ndarray.flatten(ppaGrpData)

        # the current subject's PPA mask (Sengupta et al., 2016)
        ppa_fpathes = find_files(PPA_MASK_PATTERN.replace('sub-*', subj))
        ppaIndData = fmri_dataset(ppa_fpathes,
                                  mask=occTempMaskData).samples.sum(axis=0)
        zmaps_df['ppa_ind'] = np.ndarray.flatten(ppaIndData)

        # the current subjects's AO thresholded zmap
        # get the thresholded z-maps
        aoThreshFpath = aoFpath.replace('cope1.feat/stats/zstat1.nii.gz',
                                        'cope1.feat/thresh_zstat1.nii.gz')

        if not os.path.exists(aoThreshFpath):
            subprocess.call(['datalad', 'get', aoThreshFpath])

        aoIndData = fmri_dataset(aoThreshFpath,
                                  mask=occTempMaskData).samples.sum(axis=0)

        zmaps_df['ao_ind'] = np.ndarray.flatten(aoIndData)

        # process the dataframe and do the plotting
        process_df(subj, zmaps_df, outDir)

    # create the mosaic
    infile_pattern = os.path.join(outDir, 'sub-??_bland-altman.png')
    out_fpath = os.path.join(outDir, 'subjs_bland-altman.png')
    create_mosaic(infile_pattern, '3x5', out_fpath, 80)  # columns x rows
