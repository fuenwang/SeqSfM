import numpy as np
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares

def rotate(points, rot_vecs):
    """Rotate points by given rotation vectors.
    
    Rodrigues' rotation formula is used.
    """
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v

def project(points, camera_params):
    """Convert 3-D points to 2-D by projecting onto images."""
    points_proj = rotate(points, camera_params[:, :3])
    points_proj += camera_params[:, 3:6]
    points_proj = -points_proj[:, :2] / points_proj[:, 2, np.newaxis]
    f = camera_params[:, 6]
    k1 = camera_params[:, 7]
    k2 = camera_params[:, 8]
    n = np.sum(points_proj**2, axis=1)
    r = 1 + k1 * n + k2 * n**2
    points_proj *= (r * f)[:, np.newaxis]

    return points_proj

def cost(params, n_cameras, n_points, camera_indices, point_indices, points_2d):
    """Compute residuals.

    `params` contains camera parameters and 3-D coordinates.
    """
    camera_params = params[:n_cameras * 9].reshape((n_cameras, 9))
    points_3d = params[n_cameras * 9:].reshape((n_points, 3))
    points_proj = project(points_3d[point_indices], camera_params[camera_indices])

    return (points_proj - points_2d).ravel()

def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    m = camera_indices.size * 2
    n = n_cameras * 9 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)
    for s in range(9):
        A[2 * i, camera_indices * 9 + s] = 1
        A[2 * i + 1, camera_indices * 9 + s] = 1

    for s in range(3):
        A[2 * i, n_cameras * 9 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 9 + point_indices * 3 + s] = 1

    return A


class Bundle:
    def __init__(self, config):
        self._config = config

        self._camera_param = []
        self._camera_indices = []
        self._point_indices = []
        self._point_2d = []
        self._point_3d = []
    '''
    def SetCameraNum(self, ncamera):
        self._ncameras = ncamera

    def Set3DPointNum(self, npoint):
        self._npoints = npoint

    def SetObservationNum(self, nobservation):
        self._nobservations = nobservation
    '''
    def AddCamera(self, R, T, f, d1, d2):
        self._camera_param.append(np.hstack([R, T, f, d1, d2]))

    def Add2DPoint(self, camera_index, point_index, x, y):
        self._camera_indices.apppend(camera_index)
        self._point_indices.append(point_index)
        self._point_2d.append([x, y])

    def Add3DPoint(self, x, y, z):
        self._point_3d.append([x, y, z])

    def Merge(self):
        self._camera_param = np.array(self._camera_param)
        self._camera_indices = np.array(self._camera_indices)
        self._point_indices = np.arrray(self._point_indices)
        self._point_2d = np.array(self._point_2d)
        self._point_3d = np.array(self._point_3d)

        self._ncameras = self._camera_param.shape[0]
        self._npoints = self._point_3d.shape[0]

    def Run(self):
        camera_params = self._camera_param
        camera_indices = self._camera_indices
        point_indices = self._point_indices
        points_2d = self._point_2d
        points_3d = self._point_3d
        n_cameras = self._ncameras
        n_points = self._npoints

        # To solve system
        x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))
        A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)
        res = least_squares(cost, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                    args=(n_cameras, n_points, camera_indices, point_indices, points_2d))        







