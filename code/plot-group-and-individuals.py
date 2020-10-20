#!/usr/bin/env python3
'''
created on Fri June 05 2020
author: Christian Olaf Haeusler
'''

import argparse
from glob import glob
import matplotlib as mpl
import matplotlib.pyplot as plt
import nibabel as nib
from nilearn import plotting
from nilearn.image import smooth_img
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
                              label='overlap of individual PPA masks (Sengupta et al., 2016)')

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
                              label='overlap of individual PPA masks (Sengupta et al., 2016)')

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
    # to not overlapt

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


if __name__ == "__main__":
    # read command line arguments
    outPath = parse_arguments()
    # set the background of all figures (for saving) to black
    plt.rcParams['savefig.facecolor'] = 'black'
    plt.rcParams['axes.facecolor'] = 'black'
    # cleanup in case script was interupted during debugging
    plt.close()

    # create output path in case it does not exist
    os.makedirs(outPath, exist_ok=True)

    # create probability map for results of audio-only contrasts
    print('creating probabilistic maps')
    aoPPAprobFile = 'rois-and-masks/ao_ppa_prob.nii.gz'
    create_prob_maps(AO_COPE_PATTERN,
                     AO_PPA_COPES,
                     aoPPAprobFile)

    # create probability map for results of audio-visual contrasts
    avPPAprobFile = 'rois-and-masks/av_ppa_prob.nii.gz'
    create_prob_maps(AV_COPE_PATTERN,
                     AV_PPA_COPES,
                     avPPAprobFile)

    # plotting of the probability maps (left sagittal, coronal, right saggital)
    process_stability(aoPPAprobFile,
                      avPPAprobFile,
                      outPath)

    # plot of group results
    process_group_averages(outPath)

    # process subjects
    process_individuals(AUDIO_IN_PATTERN)
