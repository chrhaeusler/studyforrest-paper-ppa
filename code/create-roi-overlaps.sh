#!/bin/bash
mkdir rois-and-masks
datalad get inputs/studyforrest-data-visualrois/sub-*/rois/*.*
datalad get inputs/studyforrest-data-visualrois/sub-*/2ndlvl.gfeat/cope*.feat/stats/zstat*.nii.gz
datalad get inputs/studyforrest-data-visualrois/src/tnt/templates/grpbold3Tp2/brain.nii.gz
datalad get inputs/studyforrest-data-visualrois/src/tnt/sub-??/bold3Tp2/in_grpbold3Tp2/subj2tmpl_warp.nii.gz

cd inputs/studyforrest-data-visualrois
./code/rois2manuscript
mv *_overlap.nii.gz ../../rois-and-masks
cd ../..
