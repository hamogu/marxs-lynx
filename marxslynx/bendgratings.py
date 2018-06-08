import numpy as np


#What is phi of center?


from marxs.math.geometry import Cylinder
from marxs.math.utils import h2e


e = instrum.elements[2].elements[200]

position = h2e(e.geometry['center']) - r * h2e(e.geometry['e_x'])

#orientation:

from transforms3d.affines import decompose, compose

r = conf['focallength']

d_phi = np.arctan(conf['grating_size'] / 2 / r)

t,rot,z,s = decompose(e.geometry.pos4d)

c = Cylinder({'position': position, 'orientation': rot,
              'zoom': r, r, conf['grating_size']/2],
              'phi_lim': [-d_phi, d_phi]})
