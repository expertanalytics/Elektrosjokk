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


class IntraAnisotropy(df.Expression):
    """Interpolation of dti."""

    def __init__(self, mesh, dti_data, r2v_affine, wm_seg, wm_ras2vox, **kwargs):
        self.GRAY_CONDUCTIVITY = 3.3        # mS/cm
        self.WHITE_CONDUCTIVITY = 1.4        # mS/cm

        k = 10
        self.white_long = self.WHITE_CONDUCTIVITY*k**(2/3)
        self.white_trans = self.WHITE_CONDUCTIVITY*k**(-1/3)

        # self.dti_data = dti_data
        self.dti_data = dti_data

        # Make it symmetric
        self.dti_data.shape = (256, 256, 256, 3, 3)      # Hackish?
        self.dti_data += np.transpose(self.dti_data, (0, 1, 2, 4, 3))
        self.dti_data *= 0.5

        self.affine = r2v_affine     # ras->vox, not vox->ras

        # Whute  matter segmentation
        self.wm_seg = wm_seg
        self.wm_ras2vox = wm_ras2vox

    def eval(self, value, x):
        if not is_white_matter(self.wm_seg, self.wm_ras2vox, x):
            normalised_coordinate_matrix = np.eye(3)*self.GRAY_CONDUCTIVITY
        else:
            i, j, k = np.rint(nib.affines.apply_affine(self.affine, x).T).astype(np.int64)
            diffusion_tensor = self.dti_data[i, j, k].reshape(3, 3)
            eigen_values, eigen_vectors = np.linalg.eig(diffusion_tensor)
            sort_indices = np.argsort(eigen_values)[::-1]      # largest to smallest

            normalised_coordinate_matrix = eigen_vectors[:, sort_indices]
            normalised_coordinate_matrix /= np.linalg.norm(normalised_coordinate_matrix, axis=1)

            diag = np.array([[self.white_long, 0, 0],
                             [0, self.white_trans, 0],
                             [0, 0, self.white_trans]])
            normalised_coordinate_matrix = normalised_coordinate_matrix@diag@normalised_coordinate_matrix.T

        eigen_values, eigen_vectors = np.linalg.eig(normalised_coordinate_matrix)
        print(eigen_values)
        value[:] = normalised_coordinate_matrix.flatten()

    def value_shape(self):
        """Return dimension."""
        return (3, 3)


class ExtraAnisotropy(df.Expression):
    """Interpolation of dti."""

    def __init__(self, mesh, dti_data, r2v_affine, wm_seg, wm_ras2vox, **kwargs):
        self.GRAY_CONDUCTIVITY = 1.26         # mS/cm
        self.WHITE_CONDUCTIVITY = 2.76        # mS/cm

        k = 10
        self.white_long = self.WHITE_CONDUCTIVITY*k**(2/3)
        self.white_trans = self.WHITE_CONDUCTIVITY*k**(-1/3)

        # self.dti_data = dti_data
        self.dti_data = dti_data

        # Make it symmetric
        self.dti_data.shape = (256, 256, 256, 3, 3)      # Hackish?
        self.dti_data += np.transpose(self.dti_data, (0, 1, 2, 4, 3))
        self.dti_data *= 0.5

        self.affine = r2v_affine     # ras->vox, not vox->ras

        # Whute  matter segmentation
        self.wm_seg = wm_seg
        self.wm_ras2vox = wm_ras2vox

    def eval(self, value, x):
        if not is_white_matter(self.wm_seg, self.wm_ras2vox, x):
            normalised_coordinate_matrix = np.eye(3)*self.GRAY_CONDUCTIVITY
        else:
            i, j, k = np.rint(nib.affines.apply_affine(self.affine, x).T).astype(np.int64)
            diffusion_tensor = self.dti_data[i, j, k].reshape(3, 3)
            eigen_values, eigen_vectors = np.linalg.eig(diffusion_tensor)
            sort_indices = np.argsort(eigen_values)[::-1]      # largest to smallest

            normalised_coordinate_matrix = eigen_vectors[:, sort_indices]
            normalised_coordinate_matrix /= np.linalg.norm(normalised_coordinate_matrix, axis=1)

            diag = np.array([[self.white_long, 0, 0],
                             [0, self.white_trans, 0],
                             [0, 0, self.white_trans]])
            normalised_coordinate_matrix = normalised_coordinate_matrix@diag@normalised_coordinate_matrix.T

        eigen_values, eigen_vectors = np.linalg.eig(normalised_coordinate_matrix)
        print(eigen_values)
        value[:] = normalised_coordinate_matrix.flatten()

    def value_shape(self):
        """Return dimension."""
        return (3, 3)


class PrincipalVector(df.Expression):
    """Interpolation of longitudinal component."""

    def __init__(self, mesh, dti_data, r2v_affine, wm_seg, wm_ras2vox, **kwargs):
        self.dti_data = dti_data
        self.affine = r2v_affine     # ras->vox, not vox->ras

        # Whute  matter segmentation
        self.wm_seg = wm_seg
        self.wm_ras2vox = wm_ras2vox

    def eval(self, value, x):
        if not is_white_matter(self.wm_seg, self.wm_ras2vox, x):
            principal_component = np.array((1, 0, 0))*0
        else:
            i, j, k = np.rint(nib.affines.apply_affine(self.affine, x).T).astype(np.int64)

            diffusion_tensor = self.dti_data[i, j, k].reshape(3, 3)
            eigen_values, eigen_vectors = np.linalg.eig(diffusion_tensor)
            max_index = np.argmax(eigen_values)
            principal_component = eigen_vectors[max_index]
            principal_component /= np.linalg.norm(principal_component)
            if principal_component[0] > 0.9:
                principal_component = np.zeros(3)

        value[:] = principal_component

    def value_shape(self):
        """Return dimension."""
        return (3,)


class WM(df.SubDomain):
    def inside(self, x, on_boundary):
        return is_white_matter(self.wm_seg, self.wm_ras2vox, x)


if __name__ == "__main__":
    DATAPATH = Path.home() / "Documents/ECT-data"


    mesh = df.Mesh(str(DATAPATH /  "meshes/bergenh18/merge.xml.gz"))
    wm_img = nib.load(str(DATAPATH / "zhi/wm.seg.mgz"))
    wm_vox2ras = wm_img.header.get_vox2ras_tkr()
    wm_ras2vox = np.linalg.inv(wm_vox2ras)

    mf = df.MeshFunction("size_t", mesh, mesh.geometry().dim())
    mf.set_all(0)
    wm = WM()
    wm.wm_seg = wm_img.get_data()
    wm.wm_ras2vox = wm_ras2vox
    wm.mark(mf, 11)
    df.File(str(DATAPATH / "meshes/bergenh18/wm.xml.gz")) << mf

    dti_img = nib.load(str(DATAPATH / "zhi/test.mgz"))
    dti_inv_aff = np.linalg.inv(dti_img.header.get_vox2ras_tkr())     # ras->vox

    func_space = df.TensorFunctionSpace(mesh, "CG", 1)

    # intraanisotropy = IntraAnisotropy(mesh, dti_img.get_data(), dti_inv_aff, wm_img.get_data(), wm_ras2vox, degree=1)
    # intraanisotropy_func = df.interpolate(intraanisotropy, func_space)
    # df.File("intraanisotropy.pvd") << intraanisotropy_func
    # df.File(str(DATAPATH / "meshes/bergenh18/intraanisotropy.xml.gz")) << intraanisotropy_func

    extraanisotropy = ExtraAnisotropy(mesh, dti_img.get_data(), dti_inv_aff, wm_img.get_data(), wm_ras2vox, degree=1)
    extraanisotropy_func = df.interpolate(extraanisotropy, func_space)
    df.File("extraanisotropy.pvd") << extraanisotropy_func
    df.File(str(DATAPATH / "meshes/bergenh18/extraanisotropy.xml.gz")) << extraanisotropy_func

    # vector_func_space = df.VectorFunctionSpace(mesh, "CG", 1)
    # principal = PrincipalVector(
    #     mesh,
    #     dti_img.get_data(),
    #     dti_inv_aff,
    #     wm_img.get_data(),
    #     wm_ras2vox,
    #     degree=1
    # )
    # vector_func = df.interpolate(principal, vector_func_space)
    # df.File("al_vector.pvd") << vector_func
    # df.File("al_vector.xml.gz") << vector_func
