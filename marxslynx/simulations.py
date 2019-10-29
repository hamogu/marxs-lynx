import numpy as np
from astropy.coordinates import SkyCoord
from marxs.source import PointSource, FixedPointing


def run_monoenergetic_simulation(instrument, energy, n_photons=2e4):
    '''Run simple simulation at fixed energy'''
    mysource = PointSource(coords=SkyCoord(0., 0., unit='deg'),
                           energy=energy,
                           flux=1.)
    fixedpointing = FixedPointing(coords=SkyCoord(0., 0., unit='deg'))
    photons = mysource.generate_photons(n_photons)
    photons = fixedpointing(photons)
    photons = instrument(photons)
    return photons[np.isfinite(photons['order'])]
