import numpy as np
from scipy.optimize import minimize_scalar
from scipy.interpolate import RectBivariateSpline
from transforms3d.affines import decompose

from marxs.math.geometry import Cylinder
from marxs.math.utils import h2e, e2h, norm_vector
from marxs.utils import generate_test_photons
from marxs.optics import OrderSelector


def bend_gratings(conf, gratings, r=None):
    '''Bend gratings in a gas to follow the Rowland cirle

    Gratings are bend in one direction (the dispersion direction) only.

    Assumes that the focal point is at the origin of the coordinate system!

    Parameters
    ----------
    conf : dict
        Configuration values for Lynx. The parameters of the Rowland circle
        and the gratings are taken form there.
    gas : list
        List of grating to be bend
    '''
    for e in gratings:
        r_elem = np.linalg.norm(h2e(e.geometry['center'])) if r is None else r

        t, rot, z, s = decompose(e.geometry.pos4d)
        d_phi = np.arctan(conf['grating_size'][1] / 2 / r_elem)
        c = Cylinder({'position': t - r_elem * h2e(e.geometry['e_x']),
                      'orientation': rot,
                      'zoom': [r_elem, r_elem, conf['grating_size'][0] / 2],
                      'phi_lim': [-d_phi, d_phi]})
        c._geometry = e.geometry._geometry
        e.geometry = c
        e.display['shape'] = 'surface'


class NumericalChirpFinder():
    uv = [0, 0]

    def __init__(self, detector, grat, order, energy, d=0.0002):
        self.photon = generate_test_photons(1)
        self.detector = detector
        self.grat = grat
        self.energy = energy
        self.order = order
        self.base_d = d
        self.calc_goal()

    def set_grat(self, grat):
        self.grat = grat
        self.calc_goal()

    def set_uv(self, uv):
        self.uv = uv
        self.init_photon()

    def calc_goal(self):
        if not hasattr(self.grat, 'original_orderselector'):
            self.grat.original_orderselector = self.grat.order_selector
        self.grat.order_selector = OrderSelector([self.order])
        self.grat._d = self.base_d
        self.set_uv([0., 0.])
        self.run_photon()
        self.goal = self.photon['detcirc_phi'][0]

    def init_photon(self):
        posongrat = h2e(self.grat.geometry['center'] +
                        self.uv[0] * self.grat.geometry['v_y'] +
                        self.uv[1] * self.grat.geometry['v_z'])
        self.pos = e2h(1.1 * posongrat, 1)
        self.dir = norm_vector(- e2h(posongrat.reshape(1, 3), 0))
        self.reset_photon()

    def reset_photon(self):
        self.photon['pos'] = self.pos
        self.photon['dir'] = self.dir
        self.photon['probability'] = 1

    def run_photon(self):
        self.reset_photon()
        self.photon = self.grat(self.photon)
        self.photon = self.detector(self.photon)

    def optimize_func(self, d):
        self.grat._d = d * self.base_d
        self.run_photon()
        return np.abs(self.photon['detcirc_phi'][0] - self.goal)

    def correction_on_d(self, uarray=np.array([-.999, 0, .999]),
                        varray=np.array([0])):
        corr = np.ones((len(uarray), len(varray)))
        for j, u in enumerate(uarray):
            for k, v in enumerate(varray):
                self.set_uv([u, v])
                corr[j, k] = minimize_scalar(self.optimize_func,
                                             bracket=(.99, 1., 1.01)).x
        return corr


def chirp_gratings(gratings, optimizer, d):
    '''
    Dimensions of corr are hardcoded
    '''
    uarray = np.array([-.999, 0, .999])
    varray = np.array([0])
    for grat in gratings:
        optimizer.set_grat(grat)
        corr = optimizer.correction_on_d(uarray, varray)
        corr = np.tile(corr[:, 0], (3, 1)).T
        ly = np.linalg.norm(grat.geometry['v_y'])
        lz = np.linalg.norm(grat.geometry['v_z'])
        grat.fitted_d_corr = corr
        grat.spline = RectBivariateSpline(ly * uarray, lz * uarray,
                                          d * corr,
                                          bbox=[-ly, ly, -lz, lz],
                                          kx=2, ky=2)

        def func(self, intercoos):
            return self.spline(intercoos[:, 0], intercoos[:, 1], grid=False)

        # Invoking the descriptor protocol to create a bound method
        # see https://stackoverflow.com/questions/972/adding-a-method-to-an-existing-object-instance
        grat._d = func.__get__(grat)
        grat.order_selector = grat.original_orderselector
