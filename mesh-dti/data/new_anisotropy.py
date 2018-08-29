import numpy as np
import nibabel as nib
import dolfin as df


class Anisotropy(df.Expression):
    """Rounding interpolation of dti."""

    def __init__(self, mesh, dti_data, r2v_affine, **kwargs):
        self.dti_data = dti_data
        self.affine = r2v_affine     # ras->vox, not vox->ras

    def eval(self, value, x):
        i, j, k = np.rint(nib.affines.apply_affine(self.affine, x).T).astype(np.int64)

        diff_tensor = self.dti_data[i, j, k]

        # if np.linalg.norm(diff_tensor) < 1e-6:
        #     diff_tensor = np.eye(3).flatten()
        # diff_tensor = np.eye(3).flatten()
        value[:] = diff_tensor

    def value_shape(self):
        """Return dimension."""
        return (3, 3)


def map_tensor_to_dofs(function_space, mgz_image):
    inv_aff = np.linalg.inv(img.get_header().get_vox2ras_tkr())     # ras->vox

    data = img.get_data()

    # Whatare theficcerent dimsnions
    dim1, dim2, dim3, dim4 = data.shape

    xyz = function_space.tabulate_dof_coordinates().reshape((function_space.dim(), -1))

    i, j, k = np.rint(apply_affine(inv_aff, xyz).T)
    values = np.zeros((len(i), dim4))
    for m in range(dim4):
        for n in range(len(i)):
            values[n, m] = data[:, :, :, m]
    return values


if __name__ == "__main__":
    img = nib.load("test.mgz")
    inv_aff = np.linalg.inv(img.get_header().get_vox2ras_tkr())     # ras->vox

    mesh = df.Mesh("merge.xml.gz")
    func_space = df.TensorFunctionSpace(mesh, "CG", 1)

    anisotropy = Anisotropy(mesh, img.get_data(), inv_aff, degree=1)
    df.File("anisotropy_new.pvd") << df.interpolate(anisotropy, func_space)
