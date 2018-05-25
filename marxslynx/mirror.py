import numpy as np
from copy import deepcopy
from scipy.interpolate import interp1d
from astropy.table import Table
from astropy.utils.data import get_pkg_data_filename

from marxs import optics
from marxs.simulator import Sequence

metashelldata = Table.read(get_pkg_data_filename('data/metashellgeom.dat',
                                                 package='marxslynx'),
                           format='ascii.ecsv')


class MetaShellAperture(optics.MultiAperture):

    def __init__(self, **kwargs):
        elements = [optics.CircleAperture(position=[10200, 0, 0],
                                          zoom=[1, shell['r_outer'], shell['r_outer']],
                                          r_inner=shell['r_inner'],
                                          id_num=shell['Metashell Serial Number'])
                    for shell in metashelldata]
        kwargs['elements'] = elements
        kwargs['id_col'] = 'shell'
        super(MetaShellAperture, self).__init__(**kwargs)
        disp = {'color': (0.0, 0.75, 0.75),
                'opacity': 0.3,
                'shape': 'triangulation',
                'outer_factor': 1.3,
                'inner_factor': 1.}
        for i in range(len(self.elements) - 1):
            self.elements[i].display = deepcopy(disp)
            self.elements[i].display['outer_factor'] = 1.0 * metashelldata['r_inner'][i + 1] / metashelldata['r_outer'][i]
        self.elements[0].display['inner_factor'] = 0
        self.elements[-1].display = deepcopy(disp)
        self.elements[-1].display['outer_factor'] = 1.5

class MetaShellMirror(optics.FlatStack):

    def __init__(self, conf):
        kwargs = {}
        kwargs['position'] = [10000, 0, 0]
        kwargs['zoom'] = [20, np.max(metashelldata['r_outer'] * 1.1),
                          np.max(metashelldata['r_outer'] * 1.1)]
        kwargs['elements'] = [optics.PerfectLens,
                              optics.RadialMirrorScatter]
        kwargs['keywords'] = [{'focallength': 10000},
                              {'inplanescatter': conf['inplanescatter'],
                               'perpplanescatter': conf['perpplanescatter']}]

        super(MetaShellMirror, self).__init__(**kwargs)


class MetaShellEfficiency(optics.base.OpticalElement):

    def __init__(self):
        data = metashelldata
        self.energies = np.array([float(i) for i in data.colnames[3:]])
        aeff = np.stack([data[c].data for c in data.colnames[3:]])
        # factor 100 is to transform cm^2 to mm^2
        self.relative_eff = aeff * 100 / (np.pi * (data['r_outer']**2 - data['r_inner']**2))
        self.shells = [(i, interp1d(self.energies, self.relative_eff[:, i - 1]))
                       for i in data['Metashell Serial Number']]

    def __call__(self, photons):
        for i, func in self.shells:
            ind = photons['shell'] == i
            prob = func(photons['energy'][ind])
            # Risk of extrapolation
            prob = np.clip(prob, 0., 1.)
            photons['probability'][ind] *= prob
        return photons


class MetaShell(Sequence):
    # pass all kwargs to MetaShellMirror.
    # This could be improved (e.g. to allow post_process_steps)
    # but not needed right now.
    def __init__(self, conf):
        super(MetaShell, self).__init__(elements=[MetaShellAperture(),
                                                  MetaShellMirror(conf),
                                                  MetaShellEfficiency()])
