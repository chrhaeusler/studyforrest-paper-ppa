#!/usr/bin/env python3
'''
created on Fri April 30 2020
author: Christian Olaf Haeusler
'''
from glob import glob
import argparse
import os.path


no2alpha = {0: 'All',
            1: 'I',
            2: 'II',
            3: 'III',
            4: 'IV',
            5: 'V',
            6: 'VI',
            7: 'VII',
            8: 'VIII'
            }


def parse_arguments():
    '''
    '''
    parser = argparse.ArgumentParser(
        description='Counts events in EV3 files for every regressor'
    )
    parser.add_argument('-d',
                        default='./events/onsets/',
                        help='the directory that contains the segmented annotation')

    parser.add_argument('-o',
                        default=None,
                        help='the output file. e.g. ./descr-stats-ao-events.tex')

    args = parser.parse_args()

    inDir = args.d
    outFile = args.o

    return inDir, outFile


if __name__ == "__main__":
    inDir, outFile = parse_arguments()

    # search for event files in the given directory
    pattern = os.path.join(inDir, '**/*.txt')
    fPathes = sorted(glob(pattern, recursive=True))

    if 'aomovie' in inDir:
        stim = 'ao'
    elif 'avmovie' in inDir:
        stim = 'av'

    # get a list of regressors from the event files
    fileNames = [os.path.basename(path) for path in fPathes]
    regressors = [os.path.splitext(fileName)[0] for fileName in fileNames]
    regressors = sorted(list(set(regressors)))

    toWrite = []
    for regressor in regressors:
        # filter list of all event files for the current loop's regressor
        pattern = '/%s.txt' % regressor
        regFiles = [ev3File for ev3File in fPathes if pattern in ev3File]

        # count the lines (=events) per file/run
        eventsPerRun = []
        for regFile in regFiles:
            noOfEvents = sum(1 for eventsPerRun in open(regFile))
            eventsPerRun.append(noOfEvents)

        # put the count for all events at the beginning of the list
        eventsPerRun.insert(0, sum(eventsPerRun))

        # build the lines for the current regressor to write to file
        for run, count in enumerate(eventsPerRun):
            regressor = regressor.capitalize()
            run = no2alpha[run]
            line = '\\newcommand{\\%s%s%s}{%s}\n' % (stim, regressor, run, count)
            toWrite.append(line)
            if run == 'VIII':
                toWrite.append('\n')

    # write the file if a filename was passed as command line argument
    toWrite = [line.replace('_', '') for line in toWrite]
    toWrite = [line.replace('-', '') for line in toWrite]
    if outFile is not None:
       with open(outFile, 'w') as f:
           f.writelines(toWrite)
