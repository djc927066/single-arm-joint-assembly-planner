"""Read the original SEALP single-table workspace instead of duplicating it."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import numpy as np
import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / 'sealp/config/sample_config.yaml'


@dataclass
class Workspace:
    table_size: np.ndarray
    table_center: np.ndarray
    table_rgba: np.ndarray
    robot_pos: np.ndarray
    robot_rot: np.ndarray
    source: str
    sha256: str

    @property
    def top_z(self):
        return float(self.table_center[2] + self.table_size[2] / 2)

    def metadata(self):
        return dict(source_config=self.source, source_sha256=self.sha256,
                    table_size=self.table_size.tolist(), table_center=self.table_center.tolist(),
                    table_top_z=self.top_z, robot_base_pos=self.robot_pos.tolist(),
                    robot_base_rotmat=self.robot_rot.tolist(), home_joints=[0.]*6,
                    collision_table='solid box from SEALP config',
                    visual_table='SEALP show_initial_layout_from_file perforated mesh and four legs')


def load_workspace(path=DEFAULT_CONFIG):
    path = Path(path).resolve()
    raw = path.read_bytes()
    data = yaml.safe_load(raw)
    robot = data['robot']
    if robot.get('type') != 'panthera_ht' or robot.get('mode') != 'single':
        raise ValueError('This entry requires the SEALP single Panthera workspace')
    obstacles = data['environment']['obstacles']
    if len(obstacles) != 1 or obstacles[0].get('name') != 'work_table' or obstacles[0].get('type') != 'box':
        raise ValueError('This Bridge entry supports exactly the configured work_table; extra obstacles need an adapter')
    table = obstacles[0]
    if not np.allclose(table.get('rotmat', np.eye(3)), np.eye(3)):
        raise ValueError('Rotated tables are not supported by the current staging sampler')
    pos, rot = np.asarray(robot['pos'], float), np.asarray(robot['rotmat'], float)
    if not np.allclose(pos, 0.) or not np.allclose(rot, np.eye(3)):
        raise ValueError('Exported cuRobo model is rooted at world origin; re-export it before changing robot base')
    size, center = np.asarray(table['extent'], float), np.asarray(table['pos'], float)
    if size.shape != (3,) or center.shape != (3,) or not np.isfinite(np.r_[size, center]).all() or (size <= 0).any():
        raise ValueError('Invalid work_table geometry')
    return Workspace(size, center, np.asarray(table.get('rgba', [.55,.45,.35,.8]), float),
                     pos, rot, str(path), hashlib.sha256(raw).hexdigest())


def bridge_site(workspace, base_vertices, requested_x=None, margin=.01):
    """Keep the entire fixed base on the table; never silently clip an explicit input."""
    v = np.asarray(base_vertices, float)
    lo = workspace.table_center[:2] - workspace.table_size[:2] / 2 + margin
    hi = workspace.table_center[:2] + workspace.table_size[:2] / 2 - margin
    min_xy, max_xy = v[:,:2].min(0), v[:,:2].max(0)
    xmin, xmax = lo[0] - min_xy[0], hi[0] - max_xy[0]
    if xmin > xmax or min_xy[1] < lo[1]-1e-8 or max_xy[1] > hi[1]+1e-8:
        raise ValueError('Bridge base cannot fit the configured table with the requested margin')
    x = float(np.clip(.4, xmin, xmax)) if requested_x is None else float(requested_x)
    if not np.isfinite(x) or x < xmin-1e-8 or x > xmax+1e-8:
        raise ValueError(f'--site-x={x} puts the Bridge base outside the table; allowed [{xmin:.4f}, {xmax:.4f}] m')
    return np.array([x, 0., workspace.top_z])
