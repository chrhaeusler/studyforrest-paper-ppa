#!/usr/bin/python3
"""
author: Christian Olaf Haeusler
created on Friday October 22th 2019
"""
import argparse
import csv
# import spacy
from collections import defaultdict


SEGMENTS_OFFSETS = (
    (0.00, 0.00),
    (886.00, 0.00),
    (1752.08, 0.08),  # third segment's start
    (2612.16, 0.16),
    (3572.20, 0.20),
    (4480.28, 0.28),
    (5342.36, 0.36),
    (6410.44, 0.44),  # last segment's start
    (7086.00, 0.00))  # movie's last time point

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
        description='Saves decriptive statistics for the annotation of speech'
    )
    parser.add_argument('-i',
                        default='annotation/fg_rscut_ad_ger_speech_tagged.tsv',
                        help='The input file')

    parser.add_argument('-o',
                        required=False,
                        default=None,
                        help='the tex-file the statistics to write to')

    args = parser.parse_args()

    inFile = args.i
    outFile = args.o

    return inFile, outFile


def read_file(inFile):
    '''
    '''
    with open(inFile) as csvfile:
        content = csv.reader(csvfile, delimiter='\t')
        header = next(content, None)
        content = [x for x in content]

    return header, content


def get_run_number(starts, onset):
    '''
    '''
    for start in sorted(starts, reverse=True):
        if float(onset) >= start:
            run = starts.index(start)
            break

    return run


def populate_nouns_count(nounsDict, data):
    '''
    '''
    segmStarts = [start for start, offset in SEGMENTS_OFFSETS]

    for line in data:
        # check the run/segment we are in
        run = get_run_number(segmStarts, line[0])
        segment = str(run + 1)

        # filter for lines that have an entry for descriptive nouns
        if len(line) >= 9 and line[9] != '':
            descrNoun = line[9].split(';')[0]
            # increment by 1 for whole stimulus [key = 0]
            nounsDict[descrNoun]['0'] += 1
            # increment by 1 for run
            nounsDict[descrNoun][segment] += 1

    # fill in a count 0 in case a run does not have any events for a category
    for noun in nounsDict.keys():
        for seg in range(0, 9):
            if str(seg) not in nounsDict[noun].keys():
                nounsDict[noun][str(seg)] = 0

    return nounsDict


def prepare_lines(nounsDict):
    '''
    '''
    linesForLatex = []
    # take slice without the category "###"
    for descrNoun in sorted(nounsDict.keys())[1:]:
        # loop trough the segments
        for segment in sorted(nounsDict[descrNoun].keys()):
            # clean the decriptive noun's category name
            name = ''.join([char for char in descrNoun if char.isalpha()])
            name = name.lower().capitalize()
            # get the count for current segment
            count = nounsDict[descrNoun][segment]
            # convert the segment number to a roman number
            segRoman = no2alpha[int(segment)]

            # create the line to be written to file
            line = '\\newcommand{\\an%s%s}{%s}\n' % (name, segRoman, count)
            linesForLatex.append(line)

        # one free line between descriptive nouns' categories
        linesForLatex.append('\n')

    return linesForLatex

# main programm
if __name__ == "__main__":
    # read the BIDS .tsv
    inFile, outFile = parse_arguments()

    # read the annotation file
    header, fContent = read_file(inFile)

    # create the dict that will contain counts of descriptive nouns per segment
    countsNouns = defaultdict(lambda: defaultdict(int))

    # populate the dict using the file's content
    countsNouns = populate_nouns_count(countsNouns, fContent)

    # prepare the lines to be written to tex-file
    linesForLatex = prepare_lines(countsNouns)

    if outFile is not None:
        with open(outFile, 'w') as f:
            f.writelines(linesForLatex)
