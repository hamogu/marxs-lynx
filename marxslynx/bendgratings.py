import numpy as np
from transforms3d.affines import decompose

from marxs.math.geometry import Cylinder
from marxs.math.utils import h2e


def bend_gratings(conf, gas, r=None):
    for e in gas.elements:
        r_elem = np.linalg.norm(h2e(e.geometry['center'])) if r is None else r

        t, rot, z, s = decompose(e.geometry.pos4d)
        d_phi = np.arctan(conf['grating_size'] / 2 / r_elem)
        c = Cylinder({'position': t - r_elem * h2e(e.geometry['e_x']),
                      'orientation': rot,
                      'zoom': [r_elem, r_elem, conf['grating_size'] / 2],
                      'phi_lim': [-d_phi, d_phi]})
        c._geometry = e.geometry._geometry
        e.geometry = c
