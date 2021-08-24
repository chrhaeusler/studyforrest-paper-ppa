#!/usr/bin/env python3
'''
created on Mon June 07 2021
author: Christian Olaf Haeusler

ToDo:
    - https://matplotlib.org/2.0.2/users/tight_layout_guide.html
    - plt.savefig(pad_inches=0) vs.  bbox_inches='tight'
    - contrained layout?

'''


import argparse
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import subprocess
from glob import glob
from nilearn import plotting
from os.path import join as opj

FS_HOME = os.environ['FREESURFER_HOME']
FG_FREESURFER = 'inputs/studyforrest-data-freesurfer'
PPA_ANA = 'inputs/studyforrest_ppa'


def parse_arguments():
    '''
    '''
    parser = argparse.ArgumentParser(
        description='Create plots using surface maps'
    )

    parser.add_argument('-i',
                        required=False,
                        default='rois-and-masks',
                        help='input folder')

    parser.add_argument('-o',
                        required=False,
                        default='paper/figures',
                        help='the folder where the figures are written into')

    args = parser.parse_args()

    inPath = args.i
    outPath = args.o

    return inPath, outPath


def find_files(pattern):
    '''
    '''
    # find the files
    found_files = glob(pattern)

    # due some sorting
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    found_files.sort(key=alphanum_key)

    return found_files


def volume_2_surf(in_file, out_path, reg, target):
    '''
    some explanation of the mri_vol2surf parameters:
        --projfrac frac : (0->1)fractional projection along normal
        --projfrac-avg min max del : average along normal
        --projfrac-max min max del : max along normal
        e.g.
        --projfrac 0.5: sample halfway between white and pial surfaces
        --projfrac-avg .2 .8 .1: start at 20%, stop at 80%, sample every 10%
    '''
    # get the filename without its path
    # rename the thresholded zmaps (output of FSL) to a readable name
    if 'audio-ppa_c1_z3.4.gfeat' in in_file:
        base = 'ao_c1_z3.4.nii.gz'
    elif 'movie-ppa_c1_z3.4.gfeat' in in_file:
        base = 'av_c1_z3.4.nii.gz'
    else:
        base = os.path.basename(in_file)

    # loop for left and right hemisphere
    for hemi in ['lh', 'rh']:
        # create output path / filename
        out_file = base.replace('.nii.gz', f'_surf_{hemi}.mgz')
        out_fpath = opj(out_path, out_file)

        # check if input file is a MASK (and thus not a z-map)
        if 'PPA' in in_file:
            # call freesurfers mri_vol2surf with smoothing
            subprocess.call(
                ['mri_vol2surf',
                 '--src', in_file,
                 '--reg', reg,  # for individuals: sub-*/mri/transforms/T2raw.dat
                 '--trgsubject', target,
                 '--hemi', hemi,
                 '--interp', 'nearest',
                 '--surf', 'white',
                 '--projfrac-max', '0', '1', '0.1',
                 '--surf-fwhm', '2',  # smoothing of x mm
                 '--o', out_fpath]
            )
        else:
            # call freesurfers mri_vol2surf without smoothing
            subprocess.call(
                ['mri_vol2surf',
                 '--src', in_file,
                 '--reg', reg,  # for individuals: sub-*/mri/transforms/T2raw.dat
                 '--trgsubject', target,
                 '--hemi', hemi,
                 '--interp', 'nearest',
                 '--surf', 'white',
                 '--projfrac-max', '0', '1', '0.1',
                 '--o', out_fpath]
            )

    return None


def create_grp_surfaces(in_path, out_path):
    '''Converts brain volume files to surface maps
    '''
    out_path = opj(out_path, 'in_t1w')
    os.makedirs(out_path, exist_ok=True)

    # input files
    in_files = dict(
        avPPAprobFile=opj(in_path, 'av_ppa_prob.nii.gz'),
        aoPPAprobFile=opj(in_path, 'ao_ppa_prob.nii.gz'),
        grpPPAbinFile=opj(in_path, 'bilat_PPA_binary.nii.gz'),
        grpPPAprobFile=opj(in_path, 'bilat_PPA_prob.nii.gz'),
        aoCope1=opj(PPA_ANA, '3rd-lvl/audio-ppa_c1_z3.4.gfeat/cope1.feat/thresh_zstat1.nii.gz'),
        avCope1=opj(PPA_ANA, '3rd-lvl/movie-ppa_c1_z3.4.gfeat/cope1.feat/thresh_zstat1.nii.gz'),
    )

    # loop over the files surface maps
    for in_file in in_files.values():
        # download input files in case they do not exist
        if not os.path.exists(in_file):
            subprocess.call(['datalad', 'get', in_file])

        # call the function
        volume_2_surf(in_file,
                      out_path,
                      opj(FS_HOME, 'average/mni152.register.dat'),
                      'MNI152_T1_1mm_brain'
                      )

    return None


def create_ind_surfaces(inPath, outPath, tmp_subjs_dir=FG_FREESURFER):
    '''
    '''
    # find all subjects for which masks exists in the input directory
    pattern = opj(inPath, 'sub-??')
    masks_pathes = find_files(pattern)
    # create a list of available subjects (as strings)
    subjs = [re.search(r'sub-..', string).group() for string in masks_pathes]
    subjs = sorted(list(set(subjs)))

    #  we will temporarily set the freesurfer subjects dir
    # from its original location to the subdataset with the data of the
    # studyforres subjects

    # get the current freesurfer subject dir
    org_subjs_dir = os.environ['SUBJECTS_DIR']

    # because mri_vol2surf doest not like the '-' in subjects' directory names
    # create a new directory and that will comprise symlinks to the subdataset
    links_dir = opj(outPath, 'freesurfer-subjects')
    os.makedirs(links_dir, exist_ok=True)
    # finally set the name of the temporal subjects dir
    os.environ['SUBJECTS_DIR'] = links_dir

    # loop over subjects for which we ran the current's paper analyses
    for subj in subjs:
        # create symbolic links that do not contain '-' in the directory name
        source = opj(tmp_subjs_dir, subj)
        destination = opj(links_dir, subj.replace('-', '0'))
        # create the link
        if not os.path.exists(destination):
            os.symlink(os.path.relpath(source, start=links_dir),
                       destination)

        # the current subject's input path
        # that contains the maps and masks in t1w
        subj_in_path = opj(inPath, subj, 'in_t1w')
        subj_masks = find_files(opj(subj_in_path, '*.nii.gz'))

        # for non-testing purposes inPath and outPath shold be the same
        # following two lines handle the testing (e.g. outPath='test')
        subj_out_path = opj(outPath, subj, 'in_t1w')
        os.makedirs(subj_out_path, exist_ok=True)

        for in_file in subj_masks:
            # loop over the two hemispheres to download inputs
            for hemi in ['lh', 'rh']:
                # define some files that will be needed
                src_registration = opj(source, 'mri/transforms', 'T2raw.dat')
                hemi_file = opj(source, 'surf', f'{hemi}.white')
                hemi_inflated = opj(source, 'surf', f'{hemi}.inflated')
                thickness = opj(source, 'surf', f'{hemi}.thickness')

                # download the files via datalad get
                for to_get in [src_registration, hemi_file, hemi_inflated, thickness]:
                    if not os.path.exists(to_get):
                        subprocess.call(['datalad', 'get', to_get])

            # call the function that calls freesurfer's mri_vol2surf
            # function will loop over hemispheres
            target = subj.replace('-', '0')
            volume_2_surf(in_file, subj_out_path, src_registration, target)

    # change Freesurfer subjects dir back to original subjects dir
    os.environ['SUBJECTS_DIR'] = org_subjs_dir

    return None


def process_grp_plotting(inPath, outPath):
    '''
    '''
    # set the inPath for maps / masks in grp t1w
    inPath = opj(inPath, 'in_t1w')

    # AO cope1
    left = opj(inPath, 'ao_c1_z3.4_surf_lh.mgz')
    right = opj(inPath, 'ao_c1_z3.4_surf_rh.mgz')
    color_map = plt.cm.get_cmap('Blues')
    color_map = color_map.reversed()
    thresh = 3.4  # usually 3.4
    out_fname = opj(outPath, f'grp_ao_c1_surf.png')

    create_2x4_plot(left, right,
                    color_map,
                    darkness=0,  # background / mesh
                    threshold=thresh,
                    vmin=thresh, vmax=5.5,
                    cbar_min=thresh, cbar_max=5.5,
                    out=out_fname
                    )

    # AV cope1
    left = opj(inPath, 'av_c1_z3.4_surf_lh.mgz')
    right = opj(inPath, 'av_c1_z3.4_surf_rh.mgz')
    color_map = plt.cm.get_cmap('YlOrRd')
    color_map = color_map.reversed()
    thresh = 3.4  # usually 3.4
    out_fname = opj(outPath, f'grp_av_c1_surf.png')

    create_2x4_plot(left, right,
                    color_map,
                    darkness=1,  # background / mesh
                    threshold=thresh,
                    vmin=thresh, vmax=5.5,
                    cbar_min=thresh, cbar_max=5.5,
                    out=out_fname
                    )

    # AO stability
    left = opj(inPath, 'ao_ppa_prob_surf_lh.mgz')
    right = opj(inPath, 'ao_ppa_prob_surf_rh.mgz')
    color_map = plt.cm.get_cmap('Blues')
    color_map = color_map.reversed()
    thresh = 1  # usually 1
    out_fname = opj(outPath, f'grp_ao_stability_surf.png')

    create_2x4_plot(left, right,
                    color_map,
                    darkness=0,  # background / mesh
                    threshold=thresh,
                    vmin=thresh, vmax=8,
                    cbar_min=thresh, cbar_max=8,
                    out=out_fname
                    )

    # AV stability
    left = opj(inPath, 'av_ppa_prob_surf_lh.mgz')
    right = opj(inPath, 'av_ppa_prob_surf_rh.mgz')
    color_map = plt.cm.get_cmap('YlOrRd')
    color_map = color_map.reversed()
    thresh = 1  # usually 1
    out_fname = opj(outPath, f'grp_av_stability_surf.png')

    create_2x4_plot(left, right,
                    color_map,
                    darkness=1,
                    threshold=thresh,
                    vmin=thresh, vmax=8,
                    cbar_min=thresh, cbar_max=8,
                    out = out_fname
                    )

    # PPA OVERLAP
    left = opj(inPath, 'bilat_PPA_prob_surf_lh.mgz')
    right = opj(inPath, 'bilat_PPA_prob_surf_rh.mgz')
    color_map = plt.cm.get_cmap('gray')  # black = small; white = high values
    thresh = 0.7   # usually 1 but we smoothed
    out_fname = opj(outPath, f'grp_ppa_surf.png')

    create_2x4_plot(left, right,
                    color_map,
                    darkness=0,  # background / mesh
                    threshold=thresh,
                    vmin=0.0001, vmax=2000000,
                    cbar_min=0.0001, cbar_max=2000000,
                    out=out_fname
                    )

    return None


def create_2x4_plot(stat_map_left, stat_map_right,
                    color_map,
                    darkness=1,
                    threshold=1,
                    vmin=1, vmax=8,
                    cbar_min=1, cbar_max=8,
                    alpha=1,
                    out='test.png'):
    '''
    '''
    # get the environment variable of the Freesurfer installation
    fs_subjs_dir = os.environ['SUBJECTS_DIR']

    # create the figure
    # fsize = (12, 8)
    fig = plt.figure(
        # figsize=fsize,
        constrained_layout=False)

    # left hemisphere; general stuff
    l_surf_mesh = opj(fs_subjs_dir, 'MNI152_T1_1mm_brain/surf/lh.inflated')
    l_backgr = opj(fs_subjs_dir, 'MNI152_T1_1mm_brain/surf/lh.sulc')

    # right hemisphere; general stuff
    r_surf_mesh = opj(fs_subjs_dir, 'MNI152_T1_1mm_brain/surf/rh.inflated')
    r_backgr = opj(fs_subjs_dir, 'MNI152_T1_1mm_brain/surf/rh.sulc')

    # left; lateral view
    ax1 = fig.add_subplot(2, 4, 1, projection='3d')
    title = None
    hemi = 'left'
    view = 'lateral'
    cbar_bool = False

    plot_surface_map(ax1,
                     color_map,
                     hemi, view,
                     l_surf_mesh, alpha,
                     l_backgr, darkness,
                     stat_map_left,
                     threshold,
                     vmin, vmax,
                     cbar_min, cbar_max,
                     cbar_bool,
                     title)

    # left; ventral view
    ax2 = fig.add_subplot(2, 4, 2, projection='3d')
    title = None  # 'Left Hemisphere'
    hemi = 'left'
    view = 'ventral'
    cbar_bool = False

    plot_surface_map(ax2,
                     color_map,
                     hemi, view,
                     l_surf_mesh, alpha,
                     l_backgr, darkness,
                     stat_map_left,
                     threshold,
                     vmin, vmax,
                     cbar_min, cbar_max,
                     cbar_bool,
                     title)

    plt.gca().invert_xaxis()

    # left; dorsal
    ax3 = fig.add_subplot(2, 4, 3, projection='3d')
    title = None  # 'Left Hemisphere'
    hemi = 'left'
    view = 'dorsal'
    cbar_bool = False

    plot_surface_map(ax3,
                     color_map,
                     hemi, view,
                     l_surf_mesh, alpha,
                     l_backgr, darkness,
                     stat_map_left,
                     threshold,
                     vmin, vmax,
                     cbar_min, cbar_max,
                     cbar_bool,
                     title)

    # right; lateral view
    ax4 = fig.add_subplot(2, 4, 4, projection='3d')
    title = None
    hemi = 'right'
    view = 'lateral'
    cbar_bool = False

    plot_surface_map(ax4,
                     color_map,
                     hemi, view,
                     r_surf_mesh, alpha,
                     r_backgr, darkness,
                     stat_map_right,
                     threshold,
                     vmin, vmax,
                     cbar_min, cbar_max,
                     cbar_bool,
                     title)

    # left; medial view
    ax5 = fig.add_subplot(2, 4, 5, projection='3d')
    title = None
    hemi = 'left'
    view = 'medial'
    cbar_bool = False

    plot_surface_map(ax5,
                     color_map,
                     hemi, view,
                     l_surf_mesh, alpha,
                     l_backgr, darkness,
                     stat_map_left,
                     threshold,
                     vmin, vmax,
                     cbar_min, cbar_max,
                     cbar_bool,
                     title)

    # right; ventral view
    ax6 = fig.add_subplot(2, 4, 6, projection='3d')
    title = None  # 'Right Hemisphere'
    hemi = 'right'
    view = 'ventral'
    cbar_bool = False

    plot_surface_map(ax6,
                     color_map,
                     hemi, view,
                     r_surf_mesh, alpha,
                     r_backgr, darkness,
                     stat_map_right,
                     threshold,
                     vmin, vmax,
                     cbar_min, cbar_max,
                     cbar_bool,
                     title)

    plt.gca().invert_xaxis()

    # right; dorsal view
    ax8 = fig.add_subplot(2, 4, 7, projection='3d')
    title = None
    hemi = 'right'
    view = 'dorsal'
    cbar_bool = False

    plot_surface_map(ax8,
                     color_map,
                     hemi, view,
                     r_surf_mesh, alpha,
                     r_backgr, darkness,
                     stat_map_right,
                     threshold,
                     vmin, vmax,
                     cbar_min, cbar_max,
                     cbar_bool,
                     title)

    # right; medial view
    ax8 = fig.add_subplot(2, 4, 8, projection='3d')
    title = None
    hemi = 'right'
    view = 'medial'
    cbar_bool = False

    plot_surface_map(ax8,
                     color_map,
                     hemi, view,
                     r_surf_mesh, alpha,
                     r_backgr, darkness,
                     stat_map_right,
                     threshold,
                     vmin, vmax,
                     cbar_min, cbar_max,
                     cbar_bool,
                     title)

    # set the space between sublots
    plt.subplots_adjust(wspace=-.20, hspace=-.68)

    # save as png
    fig.savefig(out,
                dpi=600,
                bbox_inches='tight',
                transparent=True)

#     # save as .svg
#     fig.savefig(out.replace('.png', '.svg'),
#                 dpi=600,
#                 bbox_inches='tight',
#                 transparent=True)

    plt.close()

    return None


def plot_surface_map(ax,
                     color_map,
                     hemi, view,
                     surf_mesh, alpha,
                     backgr, darkness,
                     stat_map,
                     threshold,
                     vmin, vmax,
                     cbar_vmin=1, cbar_vmax=8,
                     cbar_bool=False,
                     title=None):
    '''
    '''
    plotting.plot_surf(
        surf_mesh,
        stat_map,
        axes=ax,
        alpha=alpha,  # alpha lvl of the mesh
        bg_map=backgr,
        darkness=darkness,  # darkness of background image; 0=white
        hemi=hemi,
        view=view,
        threshold=threshold,
        vmin=threshold,
        vmax=vmax,
        cmap=color_map,
        cbar_vmin=cbar_vmin, cbar_vmax=cbar_vmax,
        colorbar=cbar_bool,
        title=title,
    )

    return ax


def process_individuals_plotting(inPath, outPath, fg_freesurfer):
    '''
    '''
    # AO cope1
    left = opj(inPath, 'SUB', 'in_t1w', 'ao-cope1-grp_surf_lh.mgz')
    right = opj(inPath, 'SUB', 'in_t1w', 'ao-cope1-grp_surf_rh.mgz')
    # colormap blue to white
    color_map = plt.cm.get_cmap('Blues')
    color_map = color_map.reversed()
    ### THRESHOLD
    thresh = 3.4  # usually 3.4
    out_fname = opj(outPath, f'subs_ao_c1_surf.png')

    create_mosaic_plot(
        fg_freesurfer,
        inPath,
        outPath,
        left, right,
        color_map,
        darkness=0,  # background / mesh
        threshold=thresh,
        vmin=thresh, vmax=7.1,
        cbar_min=thresh, cbar_max=7.1,
        out=out_fname
    )

    # AV cope1
    left = opj(inPath, 'SUB', 'in_t1w', 'av-cope1-grp_surf_lh.mgz')
    right = opj(inPath, 'SUB', 'in_t1w', 'av-cope1-grp_surf_rh.mgz')

    # colormap red to yellow
    color_map = plt.cm.get_cmap('YlOrRd')
    color_map = color_map.reversed()
    ### THRESHOLD
    thresh = 3.4  # usually 3.4
    out_fname = opj(outPath, f'subs_av_c1_surf.png')

    create_mosaic_plot(
        fg_freesurfer,
        inPath,
        outPath,
        left, right,
        color_map,
        darkness=1,  # background / mesh
        threshold=thresh,  # must be adjusted?
        vmin=thresh, vmax=7.1,
        cbar_min=thresh, cbar_max=7.1,
        out=out_fname
    )

    # individual PPA ROIS
    left = opj(inPath, 'SUB', 'in_t1w', 'PPA_?_surf_lh.mgz')
    right = opj(inPath, 'SUB', 'in_t1w', 'PPA_?_surf_rh.mgz')
    # colormap
    color_map = plt.cm.get_cmap('gray')  # black = small; white = high values
    ### THRESHOLD
    thresh = 0.7  # usually 1 but we smoothed
    out_fname = opj(outPath, f'subs_ppa_surf.png')

    create_mosaic_plot(
        fg_freesurfer,
        inPath,
        outPath,
        left, right,
        color_map,
        darkness=0,  # background / mesh
        threshold=thresh,
        vmin=0.0001, vmax=2000000,
        cbar_min=0.0001, cbar_max=2000000,
        out=out_fname
    )

    return None


def create_mosaic_plot(fg_freesurfer,
                       inPath,
                       outPath,
                       stat_map_left, stat_map_right,  # freesurfer dir
                       color_map,
                       darkness,
                       threshold,
                       vmin, vmax,
                       cbar_min, cbar_max,
                       alpha=1,
                       text=True,
                       out='test.png'):
    '''
    '''
    masks_path_pattern = opj(inPath, 'sub-??')
    masks_pathes = find_files(masks_path_pattern)

    # create a list of available subjects (as strings)
    subjs = [re.search(r'sub-..', string).group() for string in masks_pathes]
    subjs = sorted(list(set(subjs)))

    # some parameters for the figure (monster)
    cols = 4  # fixed by personal judgement
    # calculate the number of rows needed; per subject, 2 rows (=hemispheres)
    rows = 2 * (math.ceil(len(subjs) / cols))
    # fsize = (12, 12)

    # create figure
    fig = plt.figure(
        # figsize=fsize,
        constrained_layout=False
    )

    # create a list of value pairs that
    # contain the subplot's number for each subjects' hemispheres
    # DO NOT TOUCH; IT WORKS
    x = np.arange(1, cols * rows + 1)
    a = x.reshape(rows, cols)
    dunno = [np.transpose(a[x:x+2, :]) for x in list(range(0, a.shape[0], 2))]
    items = np.concatenate(dunno)

    print(f'\n{out}')
    for subj, pair in zip(subjs, items):
        # just print some feedback about the process onto command line
        print('adding', subj, 'to plot')
        # for each hemisphere of the current subject, get the subplot's number
        l_sbplt = pair[0]
        r_sbplt = pair[1]

        # list of needed files from studyforrest-data-freesurfer
        to_gets = ['lh.inflated', 'rh.inflated',
                   # 'lh.curv', 'rh.curv',
                   'lh.sulc', 'rh.sulc']
        # in case they do not exists locally, download them
        for to_get in to_gets:
            f_path = opj(fg_freesurfer, subj, 'surf', to_get)
            if not os.path.exists(to_get):
                subprocess.call(['datalad', 'get', f_path])

        # general stuff for both hemispheres
        view = 'ventral'
        alpha = 1  # of surface mesh
        cbar_bool = False
        title = None

        # left hemisphere: parameters
        hemi = 'left'
        # mesh and background
        surf_mesh = opj(fg_freesurfer, subj, 'surf', 'lh.inflated')
        backgr = opj(fg_freesurfer, subj, 'surf', 'lh.sulc')
        # map overlay

        map_left = stat_map_left.replace('SUB', subj)
        # some subjects have only one-hemispheric PPA, so search for a pattern
        if 'PPA_?_' in map_left:
            map_left = find_files(map_left)[0]

        # add the subplot of the left hemisphere to figure
        fig.add_subplot(rows, cols, l_sbplt, projection='3d')
        ax = plt.gca()

        # do the plotting
        plot_surface_map(ax,
                         color_map,
                         hemi, view,
                         surf_mesh, alpha,
                         backgr, darkness,
                         map_left,
                         threshold,
                         vmin, vmax,
                         cbar_min, cbar_max,
                         cbar_bool,
                         title)

        ax.invert_xaxis()

        # add text (the number of the subjects
        # 0, 0 is in the middle of (sub) plot (?)
        if text is True:
            ax.text2D(-0.088,  # x position; more minus -> more left
                      0.022,  # y position; more minus -> more down
                      # too high = 0.1; too low: 0.01
                      subj,  # title=subject
                      size=6,
                      color='black',
                      backgroundcolor='white',
                      # set boxcolor and its edge to white and make transparent
                      bbox=dict(facecolor=(0, 0, 0, 0),  # last number should be alpha level?
                                edgecolor=(0, 0, 0, 0))
                    )

        # right hemisphere: parameters
        hemi = 'right'
        # mesh and background
        surf_mesh = opj(fg_freesurfer, subj, 'surf', 'rh.inflated')
        backgr = opj(fg_freesurfer, subj, 'surf', 'rh.sulc')
        # map overlay
        map_right = stat_map_right.replace('SUB', subj)
        if 'PPA_?_' in map_right:
            map_right = find_files(map_right)[0]

        # add the subplot to figure
        fig.add_subplot(rows, cols, r_sbplt, projection='3d')
        ax = plt.gca()

        plot_surface_map(ax,
                         color_map,
                         hemi, view,
                         surf_mesh, alpha,
                         backgr, darkness,
                         map_right,
                         threshold,
                         vmin, vmax,
                         cbar_min, cbar_max,
                         cbar_bool,
                         title)

        ax.invert_xaxis()

    # set the space between sublots/subjects
    plt.subplots_adjust(wspace=-.45, hspace=-.65)

    # save as png
    fig.savefig(out,
                dpi=600,
                bbox_inches='tight',
                transparent=True)

#     # save as svg
#     fig.savefig(out.replace('.png', '.svg'),
#                 dpi=600
#                 bbox_inches='tight',
#                 transparent=True)

    plt.close()

    return None


def process_legend_1colorbar(outPath):
    '''
    '''
    # group, stability, audio
    plot_legend_1colorbar(
        color_first='#2474b7',
        cmap=mpl.cm.Blues.reversed(),
        label_first='descriptive nouns (8 contrasts)',
        label_second='union of individual PPA masks (Sengupta et al., 2016)',
        vmin=1,
        vmax=8,
        cbar_label='number of contrasts',
        out=opj(outPath, 'grp_ao_stability_legend.png')
    )

    # group, stability, movie
    plot_legend_1colorbar(
        color_first='#f03523',
        cmap=mpl.cm.YlOrRd.reversed(),
        label_first='movie cuts (5 contrasts)',
        label_second='union of individual PPA masks (Sengupta et al., 2016)',
        vmin=1,
        vmax=8,
        cbar_label='number of contrasts',
        out=opj(outPath, 'grp_av_stability_legend.png')
    )

    # group, primary, audio
    plot_legend_1colorbar(
        color_first='#2474b7',
        cmap=mpl.cm.Blues.reversed(),
        label_first='audio contrast \'geo, groom > all non-geo\'',
        label_second='union of individual PPA masks (Sengupta et al., 2016)',
        vmin=3.4,
        vmax=5.5,
        cbar_label='Z value',
        out=opj(outPath, 'grp_ao_c1_legend.png')
    )
    # group, primary, movie
    plot_legend_1colorbar(
        color_first='#f03523',
        cmap=mpl.cm.YlOrRd.reversed(),
        label_first='movie contrast \'vse_new > vpe_old\'',
        label_second='union of individual PPA masks (Sengupta et al., 2016)',
        vmin=3.4,
        vmax=5.5,
        cbar_label='Z value',
        out=opj(outPath, 'grp_av_c1_legend.png')
    )

    # subjects, primary, audio
    plot_legend_1colorbar(
        color_first='#2474b7',
        cmap=mpl.cm.Blues.reversed(),
        label_first='audio contrast \'geo, groom > all non-geo\'',
        label_second='individual PPA mask (Sengupta et al., 2016)',
        vmin=3.4,
        vmax=7.1,
        cbar_label='Z value',
        out=opj(outPath, 'subjs_ao_legend.png')
    )

    # subjects, primary, movie
    plot_legend_1colorbar(
        color_first='#f03523',
        cmap=mpl.cm.YlOrRd.reversed(),
        label_first='movie contrast \'vse_new > vpe_old\'',
        label_second='individual PPA mask (Sengupta et al., 2016)',
        vmin=3.4,
        vmax=7.1,
        cbar_label='Z value',
        out=opj(outPath, 'subjs_av_legend.png')
    )


def plot_legend_1colorbar(
    color_first='#2474b7',
    cmap=mpl.cm.Blues.reversed(),
    label_first='descriptive nouns (8 contrasts)',
    label_second='union of individual PPA masks (Sengupta et al., 2016)',
    vmin=1,
    vmax=8,
    cbar_label='number of contrasts',
    out='test.png'):
    '''
    '''
    fsize = (15, 6)  # width * height
    fig = plt.figure(
        figsize=fsize,
        constrained_layout=False)

    grid = fig.add_gridspec(15, 12)  # rows * columns

    # legend
    legendAxis = fig.add_subplot(grid[12:, :5])

    first = mpl.patches.Patch(color=color_first, label=label_first)
    second = mpl.patches.Patch(color='#454545', label=label_second)

    legendAxis.legend(handles=[first, second],
                      loc='center',
                      facecolor='white',  # background
                      edgecolor='k',
                      prop={'size': 12},
                      framealpha=1)

    # don't draw line on x- and y-axis
    plt.axis('off')
    # hide ticks
    #  legendAxis.tick_params(axis='both', which='both', length=0)
    # hide tick lebels
    #  plt.setp(legendAxis.get_xticklabels(), visible=False)
    #  plt.setp(legendAxis.get_yticklabels(), visible=False)

    # colorbar
    # blue colorbar for audio-description
    cax1 = fig.add_subplot(grid[12:13, 6:11])
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cb1 = mpl.colorbar.ColorbarBase(cax1,
                                    cmap=cmap,
                                    norm=norm,
                                    orientation='horizontal')

    # color edges, ticks, and tick labels and edges
    cb1.outline.set_edgecolor('k')
    cb1.set_label(cbar_label, color='k')

    cax1.tick_params(colors='k')
    if label_first == 'descriptive nouns (8 contrasts)':
        cax1.xaxis.set_ticks(list(range(1, 9)))
    elif label_first == 'movie cuts (5 contrasts)':
        cax1.xaxis.set_ticks(list(range(1, 6)))

    # shrinke the space between subplots
    plt.subplots_adjust(wspace=0, hspace=0)

    # save & close
    fname = os.path.join(out)
    plt.savefig(fname,
                dpi=200,
                bbox_inches='tight',
                pad_inches=0.2,
                transparent=True)

#     fig.savefig(fname.replace('.png', '.svg'),
#                 dpi=300,
#                 bbox_inches='tight',
#                 pad_inches=0.2,
#                 transparent=True)

    plt.close()

    return None


def process_legend_2colorbars(outPath):
    '''
    '''
    # group, stability, audio & movie
    plot_legend_2colorbars(
        blue_l='descriptive nouns (8 contrasts)',
        red_l='movie cuts (5 contrasts)',
        black_l='union of individual PPA masks (Sengupta et al., 2016)',
        vmin=1,
        vmax=8,
        label='number of contrasts',
        out=opj(outPath, 'grp_ao_av_stability_legend.png')
    )

    # group, primary, audio & movie
    plot_legend_2colorbars(
        blue_l='audio contrast \'geo, groom > all non-geo\'',
        red_l='movie contrast \'vse_new > vpe_old\'',
        black_l='union of individual PPA masks (Sengupta et al., 2016)',
        vmin=3.4,
        vmax=5.5,
        label='Z value',
        out=opj(outPath, 'grp_ao_av_c1_legend.png')
    )

    # subjects, primary, audio & movie
    plot_legend_2colorbars(
        blue_l='audio contrast \'geo, groom > all non-geo\'',
        red_l='movie contrast \'vse_new > vpe_old\'',
        black_l='individual PPA mask (Sengupta et al., 2016)',
        vmin=3.4,
        vmax=7.1,
        label='Z value',
        out=opj(outPath, 'subjs_ao_av_c1_legend.png')
    )

    return None


def plot_legend_2colorbars(
    blue_l='descriptive nouns (8 contrasts)',
    red_l='movie cuts (5 contrasts)',
    black_l='union of individual PPA masks (Sengupta et al., 2016)',
    vmin=1,
    vmax=8,
    label='number of contrasts',
    out='test.png'):
    '''
    '''

    fsize = (15, 6)  # width * height
    fig = plt.figure(
        figsize=fsize,
        constrained_layout=False)

    grid = fig.add_gridspec(15, 12)  # rows * columns

    # legend
    legendAxis = fig.add_subplot(grid[12:, :5])

    blue = mpl.patches.Patch(color='#2474b7', label=blue_l)
    red = mpl.patches.Patch(color='#f03523',  label=red_l)
    black = mpl.patches.Patch(color='#454545', label=black_l)

    legendAxis.legend(handles=[blue, red, black],
                      loc='center',
                      facecolor='white',  # background
                      edgecolor='k',
                      prop={'size': 12},
                      framealpha=1)

    # don't draw line on x- and y-axis
    plt.axis('off')
    # hide ticks
    #  legendAxis.tick_params(axis='both', which='both', length=0)
    # hide tick lebels
    #  plt.setp(legendAxis.get_xticklabels(), visible=False)
    #  plt.setp(legendAxis.get_yticklabels(), visible=False)

    # colorbar
    # blue colorbar for audio-description
    cax1 = fig.add_subplot(grid[12:13, 6:11])
    cmap = mpl.cm.Blues
    cmap = cmap.reversed()
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cb1 = mpl.colorbar.ColorbarBase(cax1,
                                    cmap=cmap,
                                    norm=norm,
                                    orientation='horizontal')

    plt.setp(cax1.get_xticklabels(), visible=False)
    cax1.tick_params(colors='k')
    cb1.outline.set_edgecolor('k')

    # red colorbar for movie
    cax2 = fig.add_subplot(grid[13:14, 6:11])
    cmap = mpl.cm.YlOrRd
    cmap = cmap.reversed()
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cb2 = mpl.colorbar.ColorbarBase(cax2,
                                    cmap=cmap,
                                    norm=norm,
                                    orientation='horizontal')

    # color edges, ticks, and tick labels and edges
    cb2.outline.set_edgecolor('k')
    if blue_l == 'descriptive nouns (8 contrasts)':
        cax2.xaxis.set_ticks(list(range(1, 9)))
    cax2.tick_params(colors='k')
    cb2.set_label(label, color='k')


    # shrinke the space between subplots
    plt.subplots_adjust(wspace=0, hspace=0)

    # save & close
    fname = os.path.join(out)
    plt.savefig(fname,
                dpi=200,
                bbox_inches='tight',
                pad_inches=0.2,
                transparent=True)

    # save as .svg
    fig.savefig(fname.replace('.png', '.svg'),
                dpi=300,
                bbox_inches='tight',
                pad_inches=0.2,
                transparent=True)

    plt.close()

    return None


if __name__ == "__main__":
    # cleanup in case script was interupted during debugging
    plt.close()

    # read command line arguments
    inPath, outPath = parse_arguments()

    # create output path in case it does not exist
    os.makedirs(outPath, exist_ok=True)

    # create surface maps in grpbold3Tp2 space
    # environment variable 'FREESURFER_HOME' must be set
    # call the function that calls freesurfer's mri_vol2surf
    print('create group surface')
    create_grp_surfaces(inPath, inPath)

    # create surface maps in individual bold3Tp2 spaces
    print('create individual surfaces')
    create_ind_surfaces(inPath, inPath)

    # plotting of surfaces in group space:
    # a) union of dedicated visualizer ROIS
    # b) primary AO & AV contrasts,
    # c) stability of AO & AV contrats
    print('processing plotting of group surfaces')
    process_grp_plotting(inPath, outPath)

    # plotting of surfaces in individual bold3Tp2 space
    # a) individual ROIs
    # b) primary AO & AV contrast
    print('processing plotting of individuals')
    process_individuals_plotting(inPath, outPath, FG_FREESURFER)

    print('processing legends and 2 colorbars')
    process_legend_2colorbars(outPath)

    # print('processing legends and 1 colorbar')
    # process_legend_1colorbar(outPath)
