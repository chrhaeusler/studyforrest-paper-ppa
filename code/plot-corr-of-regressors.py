#!/usr/bin/env python3
'''
created on Sun March 29 2020
author: Christian Olaf Haeusler
'''
from glob import glob
import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
matplotlib.use('Agg')


AO_USED = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
AO_NAMES = { 1: 'body',
             2: 'bpart',
             3: 'fahead',
             4: 'furn',
             5: 'geo',
             6: 'groom',
             7: 'object',
             8: 'se_new',
             9: 'se_old',
            10: 'sex_f',
            11: 'sex_m',
            12: 'vse_new',
            13: 'vse_old',
            14: 'vlo_ch',
            15: 'vpe_new',
            16: 'vpe_old',
            17: 'fg_ad_lrdiff',
            18: 'fg_ad_rms'
            }

AV_USED = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
AV_NAMES = { 1: 'vse_new',
             2: 'vse_old',
             3: 'vlo_ch',
             4: 'vpe_new',
             5: 'vpe_old',
             6: 'vno_cut',
             7: 'se_new (ao)',
             8: 'se_old (ao)',
             9: 'fg_av_ger_lr',
            10: 'fg_av_ger_lr_diff',
            11: 'fg_av_ger_ml',
            12: 'fg_av_ger_pd',
            13: 'fg_av_ger_rms',
            14: 'fg_av_ger_ud'
            }

def parse_arguments():
    '''
    '''
    parser = argparse.ArgumentParser(
        description="creates the correlation of convoluted regressors from \
        a subject's 1st lvl results directories (= all single run dirs ")

    parser.add_argument('-ao',
                        default='inputs/studyforrest_ppa/'\
                        'sub-01/run-1_audio-ppa-grp.feat/design.mat',
                        help='pattern of path/file for 1st lvl design files')

    parser.add_argument('-av',
                        default='inputs/studyforrest_ppa/'\
                        'sub-01/run-1_movie-ppa-grp.feat/design.mat',
                        help='pattern of path/file for 1st lvl design files')

    parser.add_argument('-o',
                        default='figures',
                        help='the output directory for the PDF and SVG file')

    args = parser.parse_args()

    aoExample = args.ao
    avExample = args.av
    outDir = args.o

    return aoExample, avExample, outDir


def plot_heatmap(matrix, outFpath):
    '''
    '''
    # generate a mask for the upper triangle
    mask = np.zeros_like(matrix, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # custom diverging colormap
    cmap = sns.diverging_palette(220, 10, sep=1, as_cmap=True)

    # draw the heatmap with the mask and correct aspect ratio
    sns_plot = sns.heatmap(matrix, mask=mask,
                           cmap=cmap,
                           square=True,
                           center=0,
                           vmin=-1.0, vmax=1,
                           annot=True, annot_kws={"size": 8}, fmt='.1f',
                           # linewidths=.5,
                           cbar_kws={"shrink": .6}
                           )

    plt.xticks(rotation=90, fontsize=12)
    plt.yticks(rotation=0, fontsize=12)

    for x in range(0, len(AO_USED)):
        plt.gca().get_xticklabels()[x].set_color('blue')  # black = default

    for x in range(len(AO_USED), len(AO_USED) + len(AV_USED)):
        plt.gca().get_xticklabels()[x].set_color('red')

    for y in range(len(AO_USED), len(AO_USED) + len(AV_USED)):
        plt.gca().get_yticklabels()[y].set_color('blue')

    for y in range(0, len(AO_USED)):
        plt.gca().get_yticklabels()[y].set_color('red')

    os.makedirs(outFpath, exist_ok=True)

    file_name = os.path.join(outFpath, 'regressor-corr.%s')
    f.savefig(file_name % 'svg', bbox_inches='tight', transparent=True)
    f.savefig(file_name % 'pdf', bbox_inches='tight', transparent=True)
    plt.close()


def find_design_files(example):
    '''
    '''
    # from example, create the pattern to find design files for all runs
    run = re.search('run-\d', example)
    run = run.group()
    designPattern = example.replace(run, 'run-*')

    # just in case, create substitute random subject for sub-01
    subj = re.search('sub-\d{2}', example)
    subj = subj.group()
    designPattern = designPattern.replace(subj, 'sub-01')

    designFpathes = sorted(glob(designPattern))

    return designFpathes


if __name__ == "__main__":
    # get the command line inputs
    aoExample, avExample, outDir = parse_arguments()

    # get design.mat files for the 8 runs of the AO & AV stimulus
    aofPathes = find_design_files(aoExample)
    avfPathes = find_design_files(avExample)

    # specify which columns of the design file to use
    # correct for python index starting at 0
    # use every 2nd column because odd numbered columns
    # in the design file are temporal derivatives
    ao_columns = [(x-1) * 2 for x in AO_USED]
    ao_reg_names = [AO_NAMES[x] for x in AO_USED]

    # do the same for the movie data
    av_columns = [(x-1) * 2 for x in AV_USED]
    av_reg_names = [AV_NAMES[x] for x in AV_USED]

    # read the all 8 design files and concatenate
    aoDf = pd.concat([pd.read_csv(run,
                                  usecols=ao_columns,
                                  names=ao_reg_names,
                                  skiprows=5, sep='\t')
                      for run in aofPathes], ignore_index=True)

    avDf = pd.concat([pd.read_csv(run,
                                  usecols=av_columns,
                                  names=av_reg_names,
                                  skiprows=5, sep='\t')
                      for run in avfPathes], ignore_index=True)

    # concatenate data of AO & AV
    all_df = pd.concat([aoDf, avDf], axis=1)

    # create the correlation matrix for all columns
    regCorrMat = all_df.corr()

    # plot it
    plot_heatmap(regCorrMat, outDir)
