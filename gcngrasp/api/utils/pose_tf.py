from __future__ import annotations

import threading

import networkx as nx
import numpy as np
import scipy

SO3 = scipy.spatial.transform.Rotation


def align_pcd(pcd1: np.ndarray,
              pcd2: np.ndarray,
              method: str,
              mask: np.ndarray = None):
    """ :return: 4x4 transformation matrix T such that $T @ pcd1 = pcd2$ """
    if mask is not None: pcd1, pcd2 = pcd1[mask], pcd2[mask]
    assert pcd1.shape[-1] == 3 and pcd1.size == pcd2.size
    T = np.eye(4, dtype=np.float32)

    if method == "linear-reg":
        import sklearn
        lm = sklearn.linear_model.LinearRegression()
        lm.fit(pcd1, pcd2)
        R, t = lm.coef_, lm.intercept_

    elif method in ("sim3", "se3"):
        # Scale factor
        centroid1, centroid2 = pcd1.mean(axis=0), pcd2.mean(axis=0)
        pcd1, pcd2 = pcd1 - centroid1, pcd2 - centroid2
        s = 1 if method == "se3" else np.sqrt(np.square(pcd2).sum() / np.square(pcd1).sum())
        # Unit quaternion
        Sxx, Sxy, Sxz, Syx, Syy, Syz, Szx, Szy, Szz = (pcd1.T @ pcd2).flatten()
        sumn = np.array([[(Sxx + Syy + Szz) / 2, Syz - Szy, Szx - Sxz, Sxy - Syx],
                         [0, (Sxx - Syy - Szz) / 2, Syx + Sxy, Szx + Sxz],
                         [0, 0, (- Sxx + Syy - Szz) / 2, Szy + Syz],
                         [0, 0, 0, (- Sxx - Syy + Szz) / 2]])
        sumn += sumn.T
        eig_val, eig_vec = np.linalg.eig(sumn)
        w, *p = eig_vec[:, eig_val.argmax()]
        # Substitute values
        R = s * SO3.from_quat([*p, w]).as_matrix()
        t = centroid2 - R @ centroid1

    else:
        raise NotImplementedError(f"Method {method} is not implemented.")
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def interp_rotation(R_src: np.ndarray,
                    R_dst: np.ndarray,
                    n: int) -> np.ndarray:
    q_src, q_dst = map(lambda R: SO3.from_matrix(R).as_quat(), (R_src, R_dst))
    qs = np.linspace(q_src, q_dst, n, endpoint=True)
    return np.stack([SO3.from_quat(q).as_matrix() for q in qs]).astype(np.float32)


def lookat_to_Tcw(pos: np.ndarray,
                  focus: np.ndarray) -> np.ndarray:
    Tcw = np.eye(4, dtype=np.float32)
    Tcw[:3, 3] = pos
    Tcw[:3, 2] = focus - pos

    Tcw[:3, 0] = np.cross(Tcw[:3, 2], np.array([0, 0, 1]))
    Tcw[:3, 1] = np.cross(Tcw[:3, 2], Tcw[:3, 0])
    Tcw[:3, :3] /= (np.linalg.norm(Tcw[:3, :3], axis=0, keepdims=True) + 1e-6)

    u, s, vt = np.linalg.svd(Tcw[:3, :3])
    Tcw[:3, :3] = u @ vt
    return Tcw


def normalize_rotation(R: np.ndarray) -> np.ndarray:
    """ Normalize a rotation matrix to ensure it is orthogonal and has determinant 1 """
    u, s, vt = np.linalg.svd(R)
    return u @ vt


def rota_toward_identity(R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ Find the rotation matrix that rotates R to identity. """
    perm = scipy.optimize.linear_sum_assignment(np.abs(R), maximize=True)[1]
    R_new = R[:, perm]
    sign = np.sign(np.diag(R_new))
    R_pcd = np.zeros((3, 3))
    R_pcd[np.arange(3), perm] = 1.
    return R_new * sign, R_pcd * sign[:, None]


class SE3:

    @classmethod
    def from_rot_trans(cls,
                       R: np.ndarray,
                       t: np.ndarray) -> np.ndarray:
        T = np.zeros([*t.shape[:-1], 4, 4], dtype=np.float32)
        T[..., :3, :3] = R
        T[..., :3, 3] = t
        T[..., -1, -1] = 1
        return T

    @classmethod
    def from_config(cls, value: str | list[float] | np.ndarray) -> np.ndarray:
        """ Parse a transformation from string or list/array to a 4x4 matrix. """
        T = np.eye(4, dtype=np.float32)
        if isinstance(value, str):
            if value not in ("eye", "identity"):
                raise ValueError(f"Invalid value: {value}")
        elif isinstance(value, (list, np.ndarray)):
            value = np.array(value, dtype=np.float32).flatten()
            size = value.size
            # tx, ty, tz, *R
            if size in (3, 6, 7):
                T[:3, 3] = value[:3]
                R: SO3 = None
                # R = roll, pitch, yaw
                if size == 6:
                    R = SO3.from_euler("XYZ", value[3:], degrees=True)
                # R = qw, qx, qy, qz
                elif size == 7:
                    R = SO3.from_quat(np.append(value[4:], value[3]))
                if R: T[:3, :3] = R.as_matrix()
            # matrix
            elif size in (12, 16):
                T[:size // 4] = value.reshape(-1, 4)
            else:
                raise ValueError(f"Invalid value size: {size}")
        else:
            raise TypeError(f"Invalid value type: {type(value)}")
        return T

    @classmethod
    def isin(cls,
             Ta: np.ndarray,
             Tb: np.ndarray,
             thresh_pos: float = .005,
             thresh_rot: float = .05) -> np.ndarray:
        Ta = Ta.reshape(-1, 4, 4)
        Tb = Tb.reshape(-1, 4, 4)
        qa = SO3.from_matrix(Ta[:, :3, :3]).as_quat()
        qb = SO3.from_matrix(Tb[:, :3, :3]).as_quat()

        ret = qa @ qb.T < thresh_rot
        ret &= np.linalg.norm(Ta[:, :3, 3][:, None] - Tb[:, :3, 3], axis=-1) < thresh_pos
        return ret.any(axis=-1)

    @classmethod
    def mean(cls, Ts: np.ndarray) -> np.ndarray:
        T = np.eye(4, dtype=np.float32)
        T[:3, 3] = Ts[:, :3, 3].mean(axis=0)
        T[:3, :3] = SO3.from_quat(SO3.from_matrix(Ts[:, :3, :3]).as_quat().mean(axis=0)).as_matrix()
        return T

    @classmethod
    def random(cls, n: int = 1):
        mat = cls.from_rot_trans(
            SO3.random(n).as_matrix(),
            np.random.randn(n, 3)
        )
        return mat if n > 1 else mat[0]

    @classmethod
    def transform_vec(cls,
                      T: np.ndarray,
                      vec: np.ndarray) -> np.ndarray:
        """ Apply rotation only to vectors. """
        return np.einsum("ij, ...j -> ...i", T[:3, :3], vec)

    @classmethod
    def transform_pcd(cls,
                      T: np.ndarray,
                      pcd: np.ndarray) -> np.ndarray:
        """ Apply full transformation (rotation + translation) to a point cloud. """
        if pcd.shape[-1] == 4:
            return np.einsum("ij, ...j -> ...i", T, pcd)
        elif pcd.shape[-1] == 3:
            return cls.transform_vec(T, pcd) + T[:3, 3]

    @classmethod
    def transform_pose(cls,
                       T: np.ndarray,
                       poses: np.ndarray) -> np.ndarray:
        """ Apply full transformation (rotation + translation) to a set of poses. """
        return np.einsum("ij, ...jk -> ...ik", T, poses)

    # tx, ty, tz, qw, qx, qy, qz
    @classmethod
    def mat2vec(cls,
                T: np.ndarray,
                scalar_first: bool = True) -> np.ndarray:
        vec = np.zeros([*T.shape[:-2], 7], dtype=np.float32)
        assert vec.ndim <= 2, "Only support single matrix or batch of matrices."
        vec[..., :3] = T[..., :3, 3]
        vec[..., 3:] = SO3.from_matrix(T[..., :3, :3]).as_quat(scalar_first=scalar_first)
        return vec

    @classmethod
    def vec2mat(cls,
                T: np.ndarray,
                scalar_first: bool = True) -> np.ndarray:
        mat = np.zeros([*T.shape[:-1], 4, 4], dtype=np.float32)
        assert mat.ndim <= 3, "Only support single vector or batch of vectors."
        mat[..., :3, 3] = T[..., :3]
        mat[..., :3, :3] = SO3.from_quat(T[..., 3:], scalar_first=scalar_first).as_matrix()
        mat[..., -1, -1] = 1
        return mat


class TFmanager:
    """
    A transformation manager that keeps track of transformations between different frames.
    """
    lock = threading.RLock()
    graph = nx.DiGraph()

    @classmethod
    def load_yaml(cls,
                  file: str):
        """
        Load transformations from a YAML file and register them.
        """
        import yaml
        with open(file, 'r') as f:
            cfg = yaml.load(f.read(), Loader=yaml.Loader)

            for dst, src_T in cfg.items():
                for src, T in src_T.items():
                    cls.register(SE3.from_config(T), src, dst)

    def __class_getitem__(cls, item) -> np.ndarray:
        """
        Usage: TFmanager[src, dst] to get the transformation matrix from `src` to `dst`.
        """
        src, dst = item
        if src == dst: return np.eye(4)

        with cls.lock:
            try:
                p = nx.shortest_path(cls.graph, src, dst)
            except (nx.exception.NetworkXNoPath, nx.exception.NodeNotFound):
                return None

            Ts = [cls.graph[p[i]][p[i + 1]]["T"] for i in range(len(p) - 1)]
            for i in range(1, len(Ts)): Ts[0] = Ts[i] @ Ts[0]
            return Ts[0]

    @classmethod
    def register(cls,
                 T: np.ndarray,
                 src: str,
                 dst: str):
        """
        Updates the internal transformation graph by registering both direct and indirect
        transformations based on transitivity.

        :param T: 4x4 transformation matrix from `src` to `dst`.
        :param src: Name of the source frame.
        :param dst: Name of the destination (parent) frame.
        """
        with cls.lock:
            assert T.shape == (4, 4), "Transformation matrix must be 4x4."
            cls.graph.add_edge(src, dst, T=T)
            cls.graph.add_edge(dst, src, T=np.linalg.inv(T))


if __name__ == '__main__':
    print(SE3.from_config([0, 0, 0.33]))

    T = SE3.random(2)
    scalar_first = False
    print(T)
    print(SE3.vec2mat(SE3.mat2vec(T, scalar_first=scalar_first), scalar_first=scalar_first))

    # Register some example transformations
    TFmanager.load_yaml("../../config/franka_tf.yaml")
    print(TFmanager.graph)

    TFmanager.register(SE3.random(), "camera", "body")
    TFmanager.register(SE3.random(), "robot", "world")
    TFmanager.register(SE3.random(), "robot", "camera")
    frame = "camera", "world"
    print(TFmanager[frame], TFmanager[frame[0], frame[1]])
