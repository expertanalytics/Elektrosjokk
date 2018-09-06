#!/usr/bin/env bash

mri_binarize --i mri/aseg.mgz --surf left_ventricle --surf-smooth 3 --match 4
mri_binarize --i mri/aseg.mgz --surf right_ventricle --surf-smooth 3 --match 43

mris_convert lh.left_ventricle left_ventricle.asc
mris_convert lh.right_ventricle right_ventricle.asc
