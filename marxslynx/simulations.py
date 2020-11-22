import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from marxs.source import PointSource, FixedPointing


def run_monoenergetic_simulation(instrument, energy, n_photons=2e4 * u.s):
    '''Run simple simulation at fixed energy'''
    mysource = PointSource(coords=SkyCoord(0., 0., unit='deg'),
                           energy=energy,
                           flux=1. / u.s / u.cm**2)
    fixedpointing = FixedPointing(coords=SkyCoord(0., 0., unit='deg'))
    photons = mysource.generate_photons(n_photons)
    photons = fixedpointing(photons)
    photons = instrument(photons)
    return photons[np.isfinite(photons['order'])]
