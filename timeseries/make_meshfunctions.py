import numpy as np
import nibabel as nib
import dolfin as df

from pathlib import Path


def is_white_matter(wm_seg, wm_ras2vox, query_point):
    """Convert from RAS to voxel coordinates and check segmemtation if white or not."""
    i, j, k = np.rint(nib.affines.apply_affine(wm_ras2vox, query_point).T).astype(np.int_)
    segmentation_value = wm_seg[i, j, k]
    return segmentation_value != 0


class WM_expression(df.Expression):

    def  __init__(self, wm_seg, wm_ras2vox, **kwargs):
        self.wm_seg = wm_seg
        self.wm_ras2vox = wm_ras2vox

    def eval(self, value, x):
        foo = is_white_matter(self.wm_seg, self.wm_ras2vox, x)
        value[:] = foo


class Anisotropy(df.Expression):
    """Rounding interpolation of dti."""

    def __init__(self, mesh, dti_data, r2v_affine, wm_seg, wm_ras2vox, **kwargs):
        self.dti_data = dti_data
        self.affine = r2v_affine     # ras->vox, not vox->ras

        # Whute  matter segmentation
        self.wm_seg = wm_seg
        self.wm_ras2vox = wm_ras2vox

    def eval(self, value, x):
        if is_white_matter(self.wm_seg, self.wm_ras2vox, x):
            diff_tensor = np.eye(3).flatten()
        else:
            i, j, k = np.rint(nib.affines.apply_affine(self.affine, x).T).astype(np.int64)

            from IPython import embed
            embed()
            diff_tensor = self.dti_data[i, j, k]
            diff_tensor /= np.repeat(
                np.linalg.norm(diff_tensor.reshape(3, 3), axis=1),
                3
            )
        value[:] = diff_tensor

    def value_shape(self):
        """Return dimension."""
        return (3, 3)


class WM(df.SubDomain):
    def inside(self, x, on_boundary):
        return is_white_matter(self.wm_seg, self.wm_ras2vox, x)


if __name__ == "__main__":
    DATAPATH = Path.home() / "Documents/ECT-data"


    mesh = df.Mesh(str(DATAPATH /  "meshes/bergenh18/merge.xml.gz"))
    wm_img = nib.load(str(DATAPATH / "zhi/wm.seg.mgz"))
    wm_vox2ras = wm_img.header.get_vox2ras_tkr()
    wm_ras2vox = np.linalg.inv(wm_vox2ras)

    # mf = df.MeshFunction("size_t", mesh, mesh.geometry().dim())
    # mf.set_all(0)
    # wm = WM()
    # wm.wm_seg = wm_img.get_data()
    # wm.wm_ras2vox = wm_ras2vox
    # wm.mark(mf, 11)

    # df.File(str(DATAPATH / "meshes/bergenh18/wm.xml.gz")) << mf

    dti_img = nib.load(str(DATAPATH / "zhi/test.mgz"))
    dti_inv_aff = np.linalg.inv(dti_img.get_header().get_vox2ras_tkr())     # ras->vox

    func_space = df.TensorFunctionSpace(mesh, "CG", 1)
    anisotropy = Anisotropy(mesh, dti_img.get_data(), dti_inv_aff, wm_img.get_data(), wm_ras2vox, degree=1)
    anisotropy_func = df.interpolate(anisotropy, func_space)
    df.File("anisotropy.pvd") << anisotropy_func
    df.File(str(DATAPATH / "meshes/bergenh18/anisotropy.xml.gz")) << anisotropy_func



    # my_point = np.array([[1, 1, 1], [2, 2, 2]])
    # img = nib.load("test.mgz")
    # inv_aff = np.linalg.inv(img.get_header().get_vox2ras_tkr())     # ras->vox

    # func_space = df.TensorFunctionSpace(mesh, "CG", 1)
    # xyz = func_space.tabulate_dof_coordinates().reshape((func_space.dim(), -1))
    # white_mask = is_white_matter(wm_img.get_data(), wm_ras2vox, xyz)
    # print("size: ", white_mask.size)
    # print(white_mask.sum())

    # anisotropy = Anisotropy(mesh, img.get_data(), inv_aff, degree=1)
    # anisotropy_func = df.interpolate(anisotropy, func_space)
    # # df.File("anisotropy_new.pvd") << anisotropy_func
    # # df.File("anisotropy_correct.xml.gz") << anisotropy_func
