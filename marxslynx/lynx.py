import copy
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.utils.data import get_pkg_data_filename
import numpy as np
import transforms3d
from scipy.interpolate import interp1d

import marxs
from marxs import optics, simulator, source, design, analysis
from marxs.design.rowland import design_tilted_torus, RowlandTorus

from . import ralfgrating
from mirror import MetaShell

from marxs import optics
from marxs.simulator import Sequence

filterdata = Table.read(get_pkg_data_filename('data/filtersqe.dat'))
filterqe = optics.GlobalEnergyFilter(filterfunc=interp1d(filterdata['energy'] / 1000,
                                                         filterdata['Total_throughput'])

# Blaze angle in degrees
blazeang = 1.91


alpha = np.deg2rad(2.2 * blazeang)
beta = np.deg2rad(4.4 * blazeang)
R, r, pos4d = design_tilted_torus(8.6e3, alpha, beta)
rowland = RowlandTorus(R, r, pos4d=pos4d)


# CAT grating
order_selector = ralfgrating.InterpolateRalfTable()

# Define L1, L2 blockage as simple filters due to geometric area
# L1 support: blocks 18 %
# L2 support: blocks 19 %
catsupport = optics.GlobalEnergyFilter(filterfunc=lambda e: 0.81 * 0.82)


class CATSupportbars(marxs.base.SimulationSequenceElement):
    '''Metal structure that holds grating facets will absorb all photons
    that do not pass through a grating facet.

    We might want to call this L3 support
    '''
    def process_photons(self, photons):
        photons['probability'][photons['facet'] < 0] = 0.
        return photons

catsupportbars = CATSupportbars()

blazemat = transforms3d.axangles.axangle2mat(np.array([0, 0, 1]),
                                             np.deg2rad(-blazeang))

gratinggrid = {'rowland': rowland, 'd_element': 72.,
               'x_range': [7e3, 9e3],
               'radius': [300, 600],
               'normal_spec': np.array([0, 0, 0, 1]),
               'elem_class': optics.CATGrating,
               'elem_args': {'d': 2e-4, 'zoom': [1., 25., 25.],
                             'orientation': blazemat,
                             'order_selector': order_selector}}

gas = design.rowland.GratingArrayStructure(**gratinggrid)

# Color gratings according to the sector they are in
for e in gas.elements:
     e.display = copy.deepcopy(e.display)
     # Angle from baseline in range 0..pi
     ang = np.arctan(e.pos4d[1,3] / e.pos4d[2, 3]) % (np.pi)
     # pick one of fixe colors
     e.display['color'] = 'rgb'[int(ang / np.pi * 3)]

det_kwargs = {'d_element': 51.,
              'elem_class': optics.FlatDetector,
              'elem_args': {'zoom': [1, 24.576, 12.288], 'pixsize': 0.024}}
det = design.rowland.RowlandCircleArray(rowland, theta=[2.3, 3.9],
**det_kwargs)

# This is just one way to establish a global coordinate system for
# detection on detectors that follow a curved surface.
# Project (not propagate) down to the focal plane.
projectfp = analysis.ProjectOntoPlane()
projectfp2 = analysis.ProjectOntoPlane()
projectfp2.loc_coos_name = ['proj2_x', 'proj2_y']

# Place an additional detector on the Rowland circle.
detcirc = optics.CircularDetector.from_rowland(rowland, width=20)
detcirc.loc_coos_name = ['detcirc_phi', 'detcirc_y']
detcirc.detpix_name = ['detcircpix_x', 'detcircpix_y']
detcirc.display['opacity'] = 0.0


target = SkyCoord(30., 30., unit='deg')
star = source.PointSource(coords=target, energy=.5, flux=1.)
pointing = source.FixedPointing(coords=target)

conf = {'inplanescatter': 2e-6,
        'perpplanescatter': 2e-6}

instrum = simulator.Sequence(elements=[MetaShell(inplanescatter=conf['inplanescatter'],
                                                 perplanescatter=conf['perpplanescatter']),
                                       gas, catsupport,
                                       catsupportbars, det, projectfp, filterqe])
