import nibabel as nib
import os

import numpy as np


def foo(fpath):
    img = nib.load(fpath)
    # print(img.get_affine())     # From voxel to scanner?
    # print(np.linalg.inv(img.get_affine()))     #  scanner_to_voxel?

    data = img.get_data()
    print(" --> shape: ", data.shape)


if __name__ == "__main__":
    fpath = "DTI1_crev_corr_regT1_RSI_minb2000_flex_res_hind_free_ih_ND.mgz"
    fpath = "DTI1_crev_corr_regT1_RSI_minb2000_flex_res_hind_free_ih_V0.mgz"
    foo(fpath)
