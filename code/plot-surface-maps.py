#!/usr/bin/env python3
'''
created on Mon June 07 2021
author: Christian Olaf Haeusler

ToDo:
    - pip install scikit-image nibabel matplotlib numpy
    - outlines of individual PPAs is not individually thresholded

'''

import argparse
from glob import glob
import matplotlib as mpl
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import os
import re
import subprocess

# underlying anatomical HD image
anatImg = '/usr/share/data/fsl-mni152-templates/MNI152_T1_0.5mm.nii.gz'

# T2* EPI group FoV
audioMask = 'rois-and-masks/fov_tmpl_0.5.nii.gz'

# PPA group masks co-registered to MNI152
PROB_MASK_GRP = 'rois-and-masks/bilat_PPA_prob.nii.gz'
BIN_MASK_GRP = 'rois-and-masks/bilat_PPA_binary.nii.gz'

# contrasts for audio-only stimulus
AO_COPE_PATTERN = 'inputs/studyforrest_ppa/3rd-lvl/'\
    'audio-ppa_c?_z3.4.gfeat/cope1.feat/thresh_zstat1.nii.gz'
# first six contrasts aim for PPA, rest are control contrasts
AO_PPA_COPES = range(1, 9)
# primary PPA contrast
AUDIO_GRP = AO_COPE_PATTERN.replace('c?', 'c1')

# contrasts of audio-visual stimulus
AV_COPE_PATTERN = 'inputs/studyforrest_ppa/3rd-lvl/'\
    'movie-ppa_c?_z3.4.gfeat/cope1.feat/thresh_zstat1.nii.gz'
# first five contrasts aim for PPA, rests are control contrasts
AV_PPA_COPES = range(1, 6)
# primary PPA CONTRAST
MOVIE_GRP = AV_COPE_PATTERN.replace('c?', 'c1')

# pattern of primary contrasts of individual subjects
AUDIO_IN_PATTERN = 'inputs/studyforrest_ppa/sub-??/'\
    '2nd-lvl_audio-ppa-grp.gfeat/cope1.feat/thresh_zstat1.nii.gz'
MOVIE_IN_PATTERN = 'inputs/studyforrest_ppa/sub-??/'\
    '2nd-lvl_movie-ppa-grp.gfeat/cope1.feat/thresh_zstat1.nii.gz'
PPA_MASK_PATTERN = 'rois-and-masks/sub-??/PPA_?_mni_bin.nii.gz'


def parse_arguments():
    '''
    '''
    parser = argparse.ArgumentParser(
        description='Create figures of results'
    )

    parser.add_argument('-o',
                        required=False,
                        default='paper/figures',
                        help='the folder where the figures are written into')

    args = parser.parse_args()

    outPath = args.o

    return outPath


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


def create_prob_maps(inFpattern, ppaCopes, outFname):
    '''
    '''
    # threshold to be used for binarizing
    zThresh = 3.4

    imageFpathes = [inFpattern.replace('?', str(x)) for x in ppaCopes]

    # initialize the image
    probData = None
    for imageFpath in imageFpathes:
        # download the image file via datalad in case it's just a sym link
        if not os.path.exists(imageFpath):
            subprocess.call(['datalad', 'get', imageFpath])

        # process first image
        if probData is None:
            threshImg = nib.load(imageFpath)
            threshData = threshImg.get_fdata()
            # binarize the data
            threshData[threshData >= zThresh] = 1
            probData = threshData
        # add other images
        else:
            threshImg = nib.load(imageFpath)
            threshData = threshImg.get_fdata()
            # binarize the data
            threshData[threshData >= zThresh] = 1
            # update the previous loops' data with current loop's data
            probData += threshData

    finalImage = nib.Nifti1Image(probData,
                                 threshImg.affine,
                                 header=threshImg.header)
    # save the image to file
    nib.save(finalImage, outFname)


def process_stability(ao_prob, av_prob, outfpath):
    '''
    '''
    print('creating plot for stability of contrasts')

    # parameters for creating the figure
    fsize = (15, 6)
    fig = plt.figure(figsize=fsize, constrained_layout=False)

    # add the grid
    grid = fig.add_gridspec(15, 12)

    # left sagittal plane
    ax1 = fig.add_subplot(grid[0:12, 0:4])
    mode = 'x'
    coord = [-28]
    # call the function that handles details of the plotting
    plot_stability_slice(mode,
                         coord,
                         BIN_MASK_GRP,
                         av_prob,
                         ao_prob,
                         axis=ax1)

    # mirror left sagittal the image such that brain 'looks' to the left
    plt.gca().invert_xaxis()

    # coronal slice
    ax2 = fig.add_subplot(grid[0:12, 4:8])
    mode = 'z'
    coord = [-11]
    # call the function that handles details of the plotting
    plot_stability_slice(mode,
                         coord,
                         BIN_MASK_GRP,
                         av_prob,
                         ao_prob,
                         axis=ax2)

    # right sagittal plane
    ax3 = fig.add_subplot(grid[0:12, 8:])
    mode = 'x'
    coord = [28]
    # call the function that handles details of the plotting
    plot_stability_slice(mode,
                         coord,
                         BIN_MASK_GRP,
                         av_prob,
                         ao_prob,
                         axis=ax3)

    # add the legend
    # plotting of the legend
    legendAxis = fig.add_subplot(grid[12:, :6])

    blue = mpl.patches.Patch(color='#2474b7',
                             label='descriptive nouns (8 contrasts)')
    red = mpl.patches.Patch(color='#f03523',
                            label='movie cuts (5 contrasts)',)
    black = mpl.patches.Patch(color='#454545',
                              label='union of individual PPA masks (Sengupta et al., 2016)')

    legendAxis.legend(handles=[blue, red, black],
                      loc='upper center',
                      facecolor='white',  # background
                      prop={'size': 12},
                      framealpha=1)

    # plotting of the colorbars
    # blue colorbar for audio-description
    cax1 = fig.add_subplot(grid[12:13, 6:11])
    cmap = mpl.cm.Blues
    cmap = cmap.reversed()
    norm = mpl.colors.Normalize(vmin=0, vmax=8)
    cb1 = mpl.colorbar.ColorbarBase(cax1,
                                    cmap=cmap,
                                    norm=norm,
                                    orientation='horizontal')

    plt.setp(cax1.get_xticklabels(), visible=False)
    cax1.tick_params(colors='w')
    cb1.outline.set_edgecolor('w')

    # red colorbar for movie
    cax2 = fig.add_subplot(grid[13:14, 6:11])
    cmap = mpl.cm.YlOrRd
    cmap = cmap.reversed()
    norm = mpl.colors.Normalize(vmin=0, vmax=8)
    cb2 = mpl.colorbar.ColorbarBase(cax2,
                                    cmap=cmap,
                                    norm=norm,
                                    orientation='horizontal')

    # ticklabels and edge of the colorbar
    cax2.tick_params(colors='w')
    cax2.xaxis.set_ticks(list(range(9)))
    cb2.set_label('number of contrasts', color='w')
    cb2.outline.set_edgecolor('w')

    # shrinke the space between subplots
    plt.subplots_adjust(wspace=0, hspace=0)

    # save & close
    fname = os.path.join(outfpath, 'stability-slices.svg')
    plt.savefig(fname,
                bbox_inches='tight',
                pad_inches=0)

    fname = os.path.join(outfpath, 'stability-slices.pdf')
    plt.savefig(fname,
                bbox_inches='tight',
                pad_inches=0)

    plt.close()

    return None


def plot_stability_slice(mode, coord,
                         roi, movie_img, audio_img,
                         axis,
                         title=None):
    '''
    '''
    # underlying MNI152 T1 0.5mm image
    colorMap = plt.cm.get_cmap('Greys')
    colorMap = colorMap.reversed()
    display = plotting.plot_anat(anat_img=anatImg,
                                 title=title,
                                 axes=axis,
                                 cut_coords=coord,
                                 display_mode=mode,
                                 cmap=colorMap,
                                 draw_cross=False)

    # add overlay of audio-description FoV
    display.add_overlay(audioMask,
                        cmap=colorMap,
                        alpha=.9)

    # add the movie contrasts on top of the anatomical image
    colorMap = plt.cm.get_cmap('YlOrRd')
    colorMap = colorMap.reversed()
    display.add_overlay(movie_img,
                        threshold=0,
                        cmap=colorMap,
                        vmin=0,
                        vmax=8,
                        alpha=1)

    # add the audio contrasts on top of the movie contrasts
    colorMap = plt.cm.get_cmap('Blues')
    colorMap = colorMap.reversed()
    display.add_overlay(audio_img,
                        threshold=0,
                        cmap=colorMap,
                        vmin=0,
                        vmax=8,
                        alpha=1)

    # add contours of group PPA
    display.add_contours(roi,
                         colors='black',
                         levels=[0.5],
                         antialiased=True,
                         linewidths=2,
                         alpha=0.55)

    return display


def process_group_averages(outfpath):
    '''
    '''
    print('creating plot for 3rd lvl group analysis')

    # parameters for creating the figure
    fsize = (15, 6)
    fig = plt.figure(figsize=fsize, constrained_layout=False)

    # add the grid
    grid = fig.add_gridspec(15, 12)

    # left sagittal plane
    ax1 = fig.add_subplot(grid[0:12, 0:4])
    mode = 'x'
    coord = [-28]
    plot_grp_slice(mode, coord,
                   BIN_MASK_GRP,
                   MOVIE_GRP,
                   AUDIO_GRP,
                   ax1)

    # mirror left sagittal the image such that brain 'looks' to the left
    plt.gca().invert_xaxis()

    plt.text(90, 110,
             'Z>3.4, p<0.05',  # title=subject
             size=15,
             color='white',
             backgroundcolor='black',
             # set boxcolor and its edge to white and make transparent
             bbox=dict(facecolor=(1, 1, 1, 0), edgecolor=(1, 1, 1, 0)))

    # plot axial / horizontal plane
    ax2 = fig.add_subplot(grid[0:12, 4:8])
    mode = 'z'
    coord = [-11]
    plot_grp_slice(mode, coord,
                   BIN_MASK_GRP,
                   MOVIE_GRP,
                   AUDIO_GRP,
                   ax2)

    # plot right sagittal plane
    ax3 = fig.add_subplot(grid[0:12, 8:])
    mode = 'x'
    coord = [28]
    plot_grp_slice(mode, coord,
                   BIN_MASK_GRP,
                   MOVIE_GRP,
                   AUDIO_GRP,
                   ax3)

    # add the legend
    # plotting of the legend
    legendAxis = fig.add_subplot(grid[12:, :6])

    blue = mpl.patches.Patch(color='#2474b7',
                             label='geo, groom > all non-geo (audio-description)')
    red = mpl.patches.Patch(color='#f03523',
                            label='vse_new > vpe_old (movie)')
    black = mpl.patches.Patch(color='#454545',
                              label='union of individual PPA masks (Sengupta et al., 2016)')

    legendAxis.legend(handles=[blue, red, black],
                      loc='upper center',
                      facecolor='white',  # white background
                      prop={'size': 12},
                      framealpha=1)

    # plotting of the colorbars
    # blue colorbar for audio-description
    cax1 = fig.add_subplot(grid[12:13, 6:11])
    cmap = mpl.cm.Blues
    cmap = cmap.reversed()
    norm = mpl.colors.Normalize(vmin=3.4, vmax=7.1)
    cb1 = mpl.colorbar.ColorbarBase(cax1,
                                    cmap=cmap,
                                    norm=norm,
                                    orientation='horizontal')

    plt.setp(cax1.get_xticklabels(), visible=False)
    cax1.tick_params(colors='w')
    cb1.outline.set_edgecolor('w')

    # red colorbar for movie
    cax2 = fig.add_subplot(grid[13:14, 6:11])
    cmap = mpl.cm.YlOrRd
    cmap = cmap.reversed()
    norm = mpl.colors.Normalize(vmin=3.4, vmax=7.1)
    cb2 = mpl.colorbar.ColorbarBase(cax2,
                                    cmap=cmap,
                                    norm=norm,
                                    orientation='horizontal')

    # ticklabels and edge of the colorbar
    cax2.tick_params(colors='w')
    cb2.set_label('Z value', color='w')
    cb2.outline.set_edgecolor('w')

    # shrinke the space between subplots
    plt.subplots_adjust(wspace=0, hspace=0)

    # save & close
    fname = os.path.join(outfpath, 'group-slices.svg')
    plt.savefig(fname,
                bbox_inches='tight',
                pad_inches=0)

    fname = os.path.join(outfpath, 'group-slices.pdf')
    plt.savefig(fname,
                bbox_inches='tight',
                pad_inches=0)
    plt.close()


def plot_grp_slice(mode, coord, roi,
                   movie_img, audio_img,
                   axis,
                   title=None):
    '''
    '''
    # underlying MNI152 T1 0.5mm image
    colorMap = plt.cm.get_cmap('Greys')
    colorMap = colorMap.reversed()
    display = plotting.plot_anat(anat_img=anatImg,
                                 title=title,
                                 axes=axis,
                                 annotate=True,  # show l/r & slice no.
                                 cut_coords=coord,
                                 display_mode=mode,
                                 cmap=colorMap,
                                 draw_cross=False)

    # add overlay of audio-description FoV
    display.add_overlay(audioMask,
                        cmap=colorMap,
                        alpha=.9)

    # add overlay of movie
    colorMap = plt.cm.get_cmap('YlOrRd')
    colorMap = colorMap.reversed()
    display.add_overlay(movie_img,
                        threshold=3.4,
                        cmap=colorMap,
                        vmin=3.4,
                        # vmax=10,
                        alpha=1)

    # add overlay of audio-only
    colorMap = plt.cm.get_cmap('Blues')
    colorMap = colorMap.reversed()
    display.add_overlay(audio_img,
                        threshold=3.4,
                        cmap=colorMap,
                        vmin=3.4,
                        # vmax=10,
                        alpha=1)

    # add contours of group PPA
    display.add_contours(roi,
                         colors='black',
                         levels=[0.5],
                         antialiased=True,
                         linewidths=2,
                         alpha=0.55)

    return display


def trim_axs(axs, N):
    '''
    '''
    axs = axs.flat

    for ax in axs[N:]:
        ax.remove()

    return axs[:N]


def process_individuals(AUDIO_IN_PATTERN):
    '''
    '''
    print('\ncreating figure with subjects as subplots')

    aoFpathes = find_files(AUDIO_IN_PATTERN)
    subjs = [re.search(r'sub-..', string).group() for string in aoFpathes]
    subjs = sorted(list(set(subjs)))

    # define the figure size
    cols = 4
    rows = len(subjs) // cols + 1
    fsize = (12, 12)

    # create 4x4 figure
    fig, axs = plt.subplots(rows, cols,
                            figsize=fsize,
                            constrained_layout=False)

    # set the space between sublots/subjects in a way that title and legend
    # to not overlap
    plt.subplots_adjust(wspace=0.0, hspace=0)
    # trim the axis using the helper function
    axs = trim_axs(axs, len(subjs))

    # loop through the data of the individuals
    for ax, subj in zip(axs, subjs):
        # prepare, process, plot and create mosaic for individuals
        print('creating subplot for', subj)
        mask_fpath = PPA_MASK_PATTERN.replace('sub-??', subj)
        mask_fpath = find_files(mask_fpath)[0]

        # plotting of statistially thresholded maps
        audio_zmap = AUDIO_IN_PATTERN.replace('sub-??', subj)
        ao_thr_map = audio_zmap  # .replace('.nii.gz', '_thresh.nii.gz')

        movie_zmap = MOVIE_IN_PATTERN.replace('sub-??', subj)
        av_thr_map = movie_zmap  # .replace('.nii.gz', '_thresh.nii.gz')

        # get the data via datalad get in case they are not downloaded yet
        if not os.path.exists(ao_thr_map):
            subprocess.call(['datalad', 'get', ao_thr_map])

        if not os.path.exists(av_thr_map):
            subprocess.call(['datalad', 'get', av_thr_map])

        # perform the plotting
        plot_thresh_zmaps(subj, ax,
                          mask_fpath, av_thr_map, ao_thr_map)

    # add legend and colorbar at right bottom of the figure
    fig = add_legend_colobar_to_individuals(fig)

    # save the plot to file
    fname = os.path.join(outPath, 'subs-thresh-ppa.svg')
    plt.savefig(fname,
                bbox_inches='tight',
                pad_inches=0)

    fname = os.path.join(outPath, 'subs-thresh-ppa.pdf')
    plt.savefig(fname,
                bbox_inches='tight',
                pad_inches=0)

    plt.close()


def plot_thresh_zmaps(subj, axis,
                      mask_fpath, av_thr_map, ao_thr_map):
    '''
    '''
    # underlying MNI152 T1 0.5mm image
    colorMap = plt.cm.get_cmap('Greys')
    colorMap = colorMap.reversed()

    display = plotting.plot_anat(anat_img=anatImg,
                                 # do not annotate but text explicitly
                                 annotate=False,
                                 # do not plot the title but text explicitly
                                 # title=title,
                                 axes=axis,
                                 display_mode='z',
                                 cut_coords=[-11],
                                 cmap=colorMap,
                                 draw_cross=False)

    plt.text(-120, +70,
             subj,  # title=subject
             size=15,
             color='white',
             backgroundcolor='black',
             # set boxcolor and its edge to white and make transparent
             bbox=dict(facecolor=(1, 1, 1, 0), edgecolor=(1, 1, 1, 0)))

    plt.text(-95, -118,  # x, y position
             'z=-11',
             size=11,
             color='white',
             backgroundcolor='black',
             # set boxcolor and its edge to white and make transparent
             bbox=dict(facecolor=(1, 1, 1, 0), edgecolor=(1, 1, 1, 0)))

    # add overlay of audio-description FoV
    display.add_overlay(audioMask,
                        cmap=colorMap,
                        alpha=.9)

    # add overlay of movie
    colorMap = plt.cm.get_cmap('YlOrRd')
    colorMap = colorMap.reversed()
    display.add_overlay(av_thr_map,
                        threshold=3.4,
                        cmap=colorMap,
                        vmin=3.4,
                        # vmax=10,
                        alpha=1)

    # add overlay of audio-only
    colorMap = plt.cm.get_cmap('Blues')
    colorMap = colorMap.reversed()
    display.add_overlay(ao_thr_map,
                        threshold=3.4,
                        cmap=colorMap,
                        vmin=3.4,
                        # vmax=10,
                        alpha=1)

    # add contours of group PPA
    # first smooth the mask
    smoothed_mask = smooth_img(mask_fpath, fwhm=2)

    display.add_contours(smoothed_mask,
                         colors='black',
                         levels=[0.5],
                         antialiased=True,
                         linewidths=2,
                         alpha=0.55)

    return None


def add_legend_colobar_to_individuals(fig):
    '''
    '''
    # make the gridspace
    # subplot of legend and subplots of colorbars share the height of one
    # brain, hence 2*4 columns = 8
    # the 3 colorbars need 3 grids, hence 8*3
    # well, it turns out, only 2 colorbars are needed
    # but spacing is cool anyway
    grid = fig.add_gridspec(8*3, 8*3)

    # plotting of the legend
    legendAxis = fig.add_subplot(grid[6*3:6*3+3, 13:23])

    blue = mpl.patches.Patch(color='#2474b7',
                             label='geo, groom > all non-geo (audio-description)')
    red = mpl.patches.Patch(color='#f03523',
                            label='vse_new > vpe_old (movie)',)
    black = mpl.patches.Patch(color='#454545',
                              label='individual PPA mask (Sengupta et al., 2016)')

    legendAxis.legend(handles=[blue, red, black],
                      loc='center',
                      facecolor='white',  # white background
                      prop={'size': 12},
                      framealpha=1)

    # plotting of the colorbars
    # blue colorbar for audio-description
    cax1 = fig.add_subplot(grid[21, 13:23])
    cmap = mpl.cm.Blues
    cmap = cmap.reversed()
    norm = mpl.colors.Normalize(vmin=3.4, vmax=7.1)
    cb1 = mpl.colorbar.ColorbarBase(cax1,
                                    cmap=cmap,
                                    norm=norm,
                                    orientation='horizontal')

    plt.setp(cax1.get_xticklabels(), visible=False)
    cax1.tick_params(colors='w')
    cb1.outline.set_edgecolor('w')

    # red colorbar for movie
    cax2 = fig.add_subplot(grid[22, 13:23])
    cmap = mpl.cm.YlOrRd
    cmap = cmap.reversed()
    norm = mpl.colors.Normalize(vmin=3.4, vmax=7.1)
    cb2 = mpl.colorbar.ColorbarBase(cax2,
                                    cmap=cmap,
                                    norm=norm,
                                    orientation='horizontal')

    # ticklabels and edge of the colorbar
    cax2.tick_params(colors='w')
    cb2.set_label('Z value', color='w')
    cb2.outline.set_edgecolor('w')

    return fig


def plot_zmaps(subj, audio_zmap, movie_zmap,
               mask_fpath, audio_thresh, movie_thresh, outfpath):
    '''
    this function was used in an early version of the script to plot individual
    unthresholded zmaps (that are still in subjects space).

    It does that by taking individual thresholds that were explored and set by
    visually inspection of unthresholded results
    e.g.:
    inputs/studyforrest_ppa/sub-01/2nd-lvl_audio-ppa-ind.gfeat/cope1.feat/stats/zstat1.nii.gz
    inputs/studyforrest_ppa/sub-01/2nd-lvl_movie-ppa-ind.gfeat/cope1.feat/stats/zstat1.nii.gz
    '''
    from nilearn.image import math_img

    THRESH_DICT = {
        'sub-01': {'AO': 2.6, 'AV': 3.4},
        'sub-02': {'AO': 2.0, 'AV': 3.2},
        'sub-03': {'AO': 2.0, 'AV': 3.2},
        'sub-04': {'AO': 3.2, 'AV': 3.0},
        'sub-05': {'AO': 2.0, 'AV': 2.4},
        'sub-06': {'AO': 2.6, 'AV': 2.4},
        'sub-09': {'AO': 2.2, 'AV': 3.2},
        'sub-14': {'AO': 2.4, 'AV': 2.4},
        'sub-15': {'AO': 3.6, 'AV': 2.8},
        'sub-16': {'AO': 3.4, 'AV': 3.4},
        'sub-17': {'AO': 3.8, 'AV': 4.0},
        'sub-18': {'AO': 3.4, 'AV': 3.0},
        'sub-19': {'AO': 3.0, 'AV': 3.6},
        'sub-20': {'AO': 3.4, 'AV': 3.2}
        }

    out_fname = os.path.join(outfpath, subj + '_ppa.svg')
    audio_thresh = THRESH_DICT[subj]['AO']
    movie_thresh = THRESH_DICT[subj]['AV']

    a_mask = math_img('img > %s' % audio_thresh, img=audio_zmap)
    azmap_masked = math_img('zmap * mask', zmap=audio_zmap, mask=a_mask)

    m_mask = math_img('img > %s' % movie_thresh, img=movie_zmap)
    mzmap_masked = math_img('zmap * mask', zmap=movie_zmap, mask=m_mask)

    # {‘ortho’, ‘tiled’, ‘x’, ‘y’, ‘z’, ‘yx’, ‘xz’, ‘yz’}
    # plotting the binary PPA mask
    display = plotting.plot_roi(mask_fpath,
                                title=subj,
                                display_mode='z',
                                cut_coords=[-11],
                                draw_cross=False,
                                cmap=plotting.cm.red_transparent,
                                vmax=15.0,
                                alpha=1)

    # add overlay of movie
    display.add_overlay(mzmap_masked,
                        threshold=movie_thresh,
                        cmap='Wistia',
                        vmin=0,
                        vmax=7,
                        alpha=.95)

    # add overlay of audio-only
    display.add_overlay(azmap_masked,
                        threshold=audio_thresh,
                        cmap=plotting.cm.black_blue,
                        vmin=0,
                        vmax=7,
                        alpha=.95)

    # save that shit
    plt.savefig(out_fname, transparent=True)
    plt.close()

    return out_fname


def pysurfer_shit():
    '''
    '''
    from surfer import Brain, project_volume_data

    subject_id = 'MNI152_T1_1mm'  # 'fsaverage' 'MNI152_T1_1mm'
    hemi = 'rh'
    surf = 'inflated'  # 'white'

    # call the Brain object constructor with theses parameters
    # to initialize the visualization session
    brain = Brain(subject_id, hemi, surf)

    # Most of the time you will be plotting data that are in MNI152 space on
    # the fsaverage brain. For this case, Freesurfer actually ships a
    # registration matrix file to align your data with the surface
    reg_file = os.path.join(os.environ['FREESURFER_HOME'],
                            'average/mni152.register.dat')

    # Note that the contours of the fsaverage surface don't perfectly match the
    # MNI brain, so this will only approximate the location of your activation
    # (although it generally does a pretty good job). A more accurate way to
    # visualize data would be to run the MNI152 brain through the recon-all pipeline.
    #
    # Alternatively, if your data are already in register with the Freesurfer
    # anatomy, you can provide project_volume_data with the subject ID, avoiding the
    # need to specify a registration file.

    # movie
    zstat = project_volume_data(avPPAprobFile, 'rh', reg_file)

    # colormap for movie
    colorMap = plt.cm.get_cmap('YlOrRd')
    colorMap = colorMap.reversed()

    # plot the movie's map
    brain.add_data(zstat, min=1, max=8, thresh=1, colormap=colorMap)

    # audio-description
    zstat = project_volume_data(aoPPAprobFile, 'rh', reg_file)

    # colormap for audio-description
    colorMap = plt.cm.get_cmap('Blues')
    colorMap = colorMap.reversed()

    # plot the audio-description's map
    brain.add_data(zstat, min=1, max=8, thresh=1, colormap=colorMap)

    # contour of GRP mask
    contour_file = BIN_MASK_GRP

    mask = project_volume_data(contour_file, 'rh', reg_file)

    brain.add_contour_overlay(mask,
                              colorbar=False,
                              min=1,
                              max=1,
                              )
                              # line_width=2)

    brain.show_view('ven')

    brain.save_imageset(subject_id, ['med', 'lat', 'ros', 'caud', 'ven'], 'png')


    return None


def edge_detection():
    '''
    '''
    # edge detection of mask
    mask_img = nib.load('rois-and-masks/bilat_PPA_binary.nii.gz')
    mask_data = mask_img.get_fdata()

    from skimage import feature
    # loop through the horizontal slices (z-coordinates)
    # and compute the edges in every 2D slices
    for z_coord in range(mask_data.shape[2]):
        # slice the 3D image
        current_slice = mask_data[:, :, z_coord]
        # get the edges
        edges = feature.canny(current_slice).astype(int)
        # put 2D images with edges into the 3D image
        mask_data[:, :, z_coord] = edges

    # just some renaming
    edge_data = mask_data

    edge_img = nib.Nifti1Image(edge_data,
                               mask_img.affine,
                               header=mask_img.header)
    # save the image to file
    edge_fpath = os.path.join(outPath, 'edge_test.nii.gz')
    nib.save(edge_img, edge_fpath)

    return


def volume_2_surf(in_file, out_path, reg, target):
    '''
    '''
    # get the filename with out path
    base = os.path.basename(in_file)

    # loop for left and right hemisphere
    for hemi in ['lh', 'rh']:
        # create output path / filename
        out_file = base.replace('.nii.gz', f'_surf_{hemi}.mgz')
        out_file = os.path.join(out_path, out_file)

        # call freesurfers mri_vol2surf
        if not os.path.exists(out_file):
            # call mri_vol2surf
            subprocess.call(['mri_vol2surf',
                                '--src', in_file,
                                '--reg', reg,
                                # '--reg', 'edit.dat',
                                '--trgsubject', target,
                                '--hemi', hemi,
                                '--interp', 'nearest',
                                '--surf', 'white',
                                #  '--surf-fwhm', '3',
                                '--o', out_file]
                            )

    return None


if __name__ == "__main__":
    from nilearn import datasets
    from nilearn import surface
    from nilearn import plotting
    from nilearn import regions

    # set the background of all figures (for saving) to black
#     plt.rcParams['savefig.facecolor'] = 'black'
#     plt.rcParams['axes.facecolor'] = 'black'

    # cleanup in case script was interupted during debugging
    plt.close()

    # read command line arguments
    outPath = parse_arguments()

    # create output path in case it does not exist
    os.makedirs(outPath, exist_ok=True)

    # create surface maps of following files in grpbold3Tp2
    avPPAprobFile = 'rois-and-masks/av_ppa_prob.nii.gz'
    aoPPAprobFile = 'rois-and-masks/ao_ppa_prob.nii.gz'
    grpPPAbinFile = 'rois-and-masks/bilat_PPA_binary.nii.gz'
    grpPPAprobFile = 'rois-and-masks/bilat_PPA_prob.nii.gz'
    in_files = [avPPAprobFile, aoPPAprobFile, grpPPAbinFile, grpPPAprobFile]

    # loop over the files surface maps
    for in_file in in_files:
        # call the function
        volume_2_surf(in_file,
                      outPath,
                      '/home/chris/freesurfer/average/mni152.register.dat',
                      'MNI152_T1_1mm_brain'
                      )

    ########################### INDIVIDUALS ##############################
    ### process individual subjects
    ROIS = 'rois-and-masks'  # = input path
    MASKS_PATH_PATTERN = os.path.join(ROIS, 'sub-??')

    masks_pathes = find_files(MASKS_PATH_PATTERN)

    # get a list of subjects (as strings)
    subjs = [re.search(r'sub-..', string).group() for string in masks_pathes]
    subjs = sorted(list(set(subjs)))

    # temporally, set freesurfer subjects dir  (e.g. ~/freesurfer/subjects)
    # to current subdatast
    # get the current dir
    ORG_FS_SUBJS_DIR = os.environ['SUBJECTS_DIR']
    # the subdataset with forrest freesurfer data
    TEMP_FS_DATA = os.path.expanduser('inputs/studyforrest-data-freesurfer')

    ### THE symlinks (cause mri_vol2surf doest not like '-'
    LINKS_DIR = os.path.join('test', 'freesurfer-subjects')
    os.makedirs(LINKS_DIR, exist_ok = True)
    os.environ['SUBJECTS_DIR'] = LINKS_DIR

    for subj in subjs[7:8]:
        # make as symbolic link to the freesurfer subject dir
        source = os.path.join(TEMP_FS_DATA, subj)
        destination = os.path.join(LINKS_DIR, subj.replace('-', '0'))

        # create the link
        if not os.path.exists(destination):
            os.symlink(os.path.relpath(source, start=LINKS_DIR),
                       destination)

        ### TEMPORARY INPATH ###
        in_path = os.path.join('test', subj, 'in_t1w')
        in_masks = find_files(os.path.join(in_path, '*.nii.gz'))

        for in_file in in_masks:
            # in_file = os.path.basename(in_file)
            # loop over the two hemispheres
            for hemi in ['lh', 'rh']:

                # define some files that will be needed
                src_registration = os.path.join(source, 'mri/transforms/T2raw.dat')
                hemi_file = os.path.join(source, 'surf', f'{hemi}.white')
                hemi_inflated = os.path.join(source, 'surf', f'{hemi}.inflated')
                # put them into a list
                to_gets = [src_registration, hemi_file, hemi_inflated]

                # download the files via datalad get
                for to_get in to_gets:
                    if not os.path.exists(to_get):
                        subprocess.call(['datalad', 'get', to_get])

                out_file = in_file.replace('.nii.gz', f'_surf-{hemi}.mgz')
                # out_file = os.path.join(outPath, out_file)

                if not os.path.exists(out_file):
                    # call mri_vol2surf
                    subprocess.call(
                        ['mri_vol2surf',
                         '--src', in_file,
                         '--reg', src_registration,
                         # '--reg', 'edit.dat',
                         '--trgsubject', subj.replace('-', '0'),
                         '--hemi', hemi,
                         '--interp', 'nearest',
                         '--surf', 'white',
                         #  '--surf-fwhm', '3',
                         '--o', out_file]
                    )
    # change Freesurfer subjects dir back to original subjects dir
    os.environ['SUBJECTS_DIR'] = ORG_FS_SUBJS_DIR

    # PLOTTING OF AO
#     colorMap = plt.cm.get_cmap('Blues')
#     colorMap = colorMap.reversed()
#
#     plotting.plot_surf_stat_map(
#         '/home/chris/freesurfer/subjects/MNI152_T1_1mm_brain/surf/rh.inflated',  # surf_mash
#         # '/home/chris/ao_ppa_prob-surf-rh.mgz',  # stat_map
#         'ao_ppa_prob_surf_rh.mgz',
#         bg_map='/home/chris/freesurfer/subjects/MNI152_T1_1mm_brain/surf/rh.sulc',  # curv vs. sulc?
#         # axes=axs,
#         # title='surface right hemisphere',
#         hemi='right',
#         view='ventral', # 'lateral',  # 'ventral', # lateral'
#         cmap=colorMap,
#         threshold=1,
#         vmax=14,
#         alpha=1.0,  # alpha lvl of the mesh
#         darkness=1,  # darkness of background image; 0=white
#         colorbar=True
#     )
#
#     # save map
#     out_fname = os.path.join(outPath, 'surface-plot-ao.png')
#     plt.savefig(out_fname) #  transparent=True)
#
#     # PLOTTING OF Mask
#     colorMap = plt.cm.get_cmap('Blues')
#     colorMap = colorMap.reversed()
#
#     plotting.plot_surf_stat_map(
#         '/home/chris/freesurfer/subjects/MNI152_T1_1mm_brain/surf/rh.inflated',  # surf_mash
#         # '/home/chris/ao_ppa_prob-surf-rh.mgz',  # stat_map
#         'bilat_PPA_prob_surf_rh.mgz',
#         bg_map='/home/chris/freesurfer/subjects/MNI152_T1_1mm_brain/surf/rh.sulc',  # curv vs. sulc?
#         # axes=axs,
#         # title='surface right hemisphere',
#         hemi='right',
#         view='ventral', # 'lateral',  # 'ventral', # lateral'
#         cmap=colorMap,
#         threshold=1,
#         vmax=14,
#         alpha=1.0,  # alpha lvl of the mesh
#         darkness=1,  # darkness of background image; 0=white
#         colorbar=True
#     )
#
#     # save map
#     out_fname = os.path.join(outPath, 'surface-plot-mask.png')
#     plt.savefig(out_fname) #  transparent=True)
#
#     # maybe use nilearn.regions.connected_regions
#
#
# #    data = surface.load_surf_data('/home/chris/bilat_PPA_binary_surf-rh.mgz')
# #     coords, faces = surface.load_surf_mesh('/home/chris/freesurfer/subjects/MNI152_T1_1mm_brain/surf/rh.inflated')
# #
# #     plotting.plot_surf_contours(
# #         '/home/chris/freesurfer/subjects/MNI152_T1_1mm_brain/surf/rh.inflated',  # surf_mash
# #         '/home/chris/rPPA_overlap_surf-rh.mgz',
# #         # figure=fig,
# #         cmap='Greens')
#
#     # save that shit
#
#
# #     out_fname = out_fname.replace('.png', '.svg')
# #     plt.savefig(out_fname, transparent=True)
#
# #    plt.show()
