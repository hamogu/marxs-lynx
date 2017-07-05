import copy
from astropy.coordinates import SkyCoord
from astropy.utils.data import get_pkg_data_filename
import numpy as np
import transforms3d

import marxs
from marxs import optics, simulator, source, design
from marxs.design import design_tilted_torus, RowlandTorus

from . import ralfgrating

# Blaze angle in degrees
blazeang = 1.91

alpha = np.deg2rad(2.2 * blazeang)
beta = np.deg2rad(4.4 * blazeang)
R, r, pos4d = design_tilted_torus(9e3, alpha, beta)
rowland = RowlandTorus(R, r, pos4d=pos4d)

aper = optics.CircleAperture(position=[9500, 0, 0],
                             zoom=[1, 600, 600],
                             r_inner=300)

# 0.5 arcsec mirrors
scatter = np.deg2rad(0.5 / 3600.)
mirr = optics.FlatStack(position=[9000, 0, 0], zoom=[20, 600, 600],
                        elements=[optics.PerfectLens,
                                  optics.RadialMirrorScatter],
                        keywords=[{'focallength': 9000},
                                  {'inplanescatter': scatter,
                                   'perpplanescatter': scatter}])


# CAT grating
order_selector = ralfgrating.InterpolateRalfTable(get_pkg_data_filename('efficiency.csv'))

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
gratquality = ralfgrating.RalfQualityFactor(d=200.e-3, sigma=1.75e-3)

gratinggrid = {'rowland': rowland, 'd_element': 55., 'x_range': [8e3, 9e3],
               'elem_class': optics.CATGrating,
               'elem_args': {'d': 2e-4, 'zoom': [1., 25., 25.],
                             'orientation': blazemat,
                             'order_selector': order_selector}}

gas = design.rowland.GratingArrayStructure(rowland=rowland,
                                           d_element=60,
                                           x_range=[7000, 9000],
                                           radius=[300, 600],
                                           elem_class=optics.CATGrating,
                                           elem_args={'d': 2e-4,
                                                      'zoom': [1, 25, 25],
                                                      'order_selector': order_selector,
                                                      'orientation':
blazemat})

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
projectfp = marxs.analysis.ProjectOntoPlane()

# Place an additional detector on the Rowland circle.
detcirc = optics.CircularDetector.from_rowland(rowland, width=20)
detcirc.loc_coos_name = ['detcirc_phi', 'detcirc_y']
detcirc.detpix_name = ['detcircpix_x', 'detcircpix_y']
detcirc.display['opacity'] = 0.0


target = SkyCoord(30., 30., unit='deg')
star = source.PointSource(coords=target, energy=.5, flux=1.)
pointing = source.FixedPointing(coords=target)
instrum = simulator.Sequence(elements=[aper, mirr, gas, det])
