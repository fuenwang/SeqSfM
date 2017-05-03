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

def project(points, camera_params, f, k1, k2):
    """Convert 3-D points to 2-D by projecting onto images."""
    points_proj = rotate(points, camera_params[:, :3])
    points_proj += camera_params[:, 3:6]
    points_proj = -points_proj[:, :2] / points_proj[:, 2, np.newaxis]
    #f = camera_params[:, 6]
    #k1 = camera_params[:, 7]
    #k2 = camera_params[:, 8]

    #f_n = np.zeros(points.shape[0], dtype=np.float) + f
    #k1_n = np.zeros(points.shape[0], dtype=np.float) + k1
    #k2_n = np.zeros(points.shape[0], dtype=np.float) + k2

    n = np.sum(points_proj**2, axis=1)
    r = 1 + k1 * n + k2 * n**2
    points_proj *= (r * f)[:, np.newaxis]
    return points_proj

def cost(params, n_cameras, n_points, camera_indices, point_indices, points_2d):
    """Compute residuals.

    `params` contains camera parameters and 3-D coordinates.
    """
    camera_params = params[3:3+n_cameras * 6].reshape((n_cameras, 6))
    points_3d = params[3+n_cameras * 6:].reshape((n_points, 3))
    points_proj = project(points_3d[point_indices], camera_params[camera_indices], params[0], params[1], params[2])
    #print points_proj[:20]
    #print points_2d[:20]
    #exit()
    a = points_proj + points_2d
    #a = points_proj - points_2d
    a = np.sqrt(a[:, 0]**2 + a[:, 1]**2)
    return a.ravel()

def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    m = camera_indices.size
    n = 3 + n_cameras * 6 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)
    A[:, :3] = 1
    for s in range(6):
        A[i, 3 + camera_indices * 6 + s] = 1

    for s in range(3):
        A[i, 3 + n_cameras * 6 + point_indices * 3 + s] = 1

    return A


class Bundle:
    def __init__(self):

        self._camera_param = []
        self._camera_indices = []
        self._point_indices = []
        self._point_2d = []
        self._point_3d = []

        self._camera_tag = {}
        self._point_tag = {}

        self._camera_start = 0
        self._point_start = 0
    '''
    def SetCameraNum(self, ncamera):
        self._ncameras = ncamera

    def Set3DPointNum(self, npoint):
        self._npoints = npoint

    def SetObservationNum(self, nobservation):
        self._nobservations = nobservation
    '''
    def SetCamera(self, f, k1, k2):
        self.f = f
        self.k1 = k1
        self.k2 = k2

    def AddCamera_tag(self, tag, R, T):
        if tag not in self._camera_tag:
            self._camera_param.append(np.hstack([R, T]))
            self._camera_tag[tag] = self._camera_start
            self._camera_start += 1

    def Add2DPoint_tag(self, camera_tag, point_tag, x, y):
        self._camera_indices.append(self._camera_tag[camera_tag])
        self._point_indices.append(self._point_tag[point_tag])
        self._point_2d.append([x, y])

    def Add3DPoint_tag(self, tag, x, y, z):
        if tag not in self._point_tag:
            self._point_3d.append([x, y, z])
            self._point_tag[tag] = self._point_start
            self._point_start += 1


    def Merge(self):
        self._camera_param = np.array(self._camera_param)
        self._camera_indices = np.array(self._camera_indices)
        self._point_indices = np.array(self._point_indices)
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
        x0 = np.hstack([[self.f, self.k1, self.k2], camera_params.ravel(), points_3d.ravel()])
        f = cost(x0, n_cameras, n_points, camera_indices, point_indices, points_2d)
        #print f
        #print np.max(f)
        #exit()
        A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)
        self.res = least_squares(cost, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-7, method='trf',
                    args=(n_cameras, n_points, camera_indices, point_indices, points_2d))        

        self.result = {}
        self.result['f'] = self.res['x'][0]
        self.result['k1'] = self.res['x'][1]
        self.result['k2'] = self.res['x'][2]

        self.result['camera_params'] = self.res['x'][3:3+n_cameras*6].reshape([n_cameras, 6])
        self.result['points_3d'] = self.res['x'][3+n_cameras*6:].reshape([n_points, 3])

    def GetShot(self, shot):
        index = self._camera_tag[shot]
        RT = self.result['camera_params'][index, :]
        return RT[:3], RT[3:]

    def GetPoint(self, track_id):
        index = self._point_tag[track_id]
        loc = self.result['points_3d'][index, :]
        return loc

    def GetCameraIntrinsic(self):
        return [self.result['f'], self.result['k1'], self.result['k2']]




