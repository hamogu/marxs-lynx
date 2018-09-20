import numpy as np
from numpy.core.umath_tests import inner1d
from scipy.interpolate import RectBivariateSpline
from transforms3d.affines import decompose
import astropy.units as u

from marxs.math.geometry import Cylinder
from marxs.math.utils import h2e, norm_vector
from marxs.utils import generate_test_photons
from marxs.optics import CATGrating

from .lynx import detcirc


def bend_gratings(conf, gas, r=None):
    for e in gas.elements:
        r_elem = np.linalg.norm(h2e(e.geometry['center'])) if r is None else r

        t, rot, z, s = decompose(e.geometry.pos4d)
        d_phi = np.arctan(conf['grating_size'][1] / 2 / r_elem)
        c = Cylinder({'position': t - r_elem * h2e(e.geometry['e_x']),
                      'orientation': rot,
                      'zoom': [r_elem, r_elem, conf['grating_size'][0] / 2],
                      'phi_lim': [-d_phi, d_phi]})
        c._geometry = e.geometry._geometry
        e.geometry = c


def find_where_ref_ray_should_go(conf, order, wave):
    energy = wave.to(u.keV, equivalencies=u.spectral())
    rays = generate_test_photons(2)
    rays['energy'] = energy.value
    rays['pos'][:, 0] = 1e5

    def mock_order(x, y, z):
        return np.array([0, order]), np.ones(2)

    pos = conf['rowland'].solve_quartic(y=0, z=0, interval=[9e3, 1.2e4])
    test_grat = CATGrating(d=conf['grating_d'], order_selector=mock_order,
                            position=[pos, 0, 0],
                            orientation=conf['blazemat'] )
    rays = test_grat(rays)
    rays = detcirc(rays)
    return rays['pos'].data[0, :], rays['pos'].data[1, :]


def chirp_flat_grating(conf, gas, order, wave, n_points=[3, 3]):
    '''
    Parameters
    ----------
    wave : `astropy.quantity.Quantity`
    '''
    focalpoint, ref_point = find_where_ref_ray_should_go(conf, order, wave)
    pos_on_e = np.meshgrid(np.linspace(-1, 1, n_points[0]), np.linspace(-1, 1, n_points[1]))
    d_needed = np.zeros_like(pos_on_e[0])
    for e in gas.elements:
        l_x = np.linalg.norm(e.geometry['v_y'])
        l_y = np.linalg.norm(e.geometry['v_z'])
        for i in range(pos_on_e[0].shape[0]):
            for j in range(pos_on_e[0].shape[1]):
                positions = h2e(e.geometry['center']) + pos_on_e[0][i, j] * h2e(e.geometry['v_y']) + pos_on_e[1][i, j] * h2e(e.geometry['v_z'])
                vec_pos_foc = - positions[:2] + h2e(focalpoint)[:2]
                vec_pos_foc = vec_pos_foc / np.linalg.norm(vec_pos_foc)
                vec_pos_ref_point = - positions[:2] + h2e(ref_point)[:2]
                vec_pos_ref_point = vec_pos_ref_point / np.linalg.norm(vec_pos_ref_point)

                theta_needed = np.arccos(np.dot(vec_pos_foc, vec_pos_ref_point))
                d_needed[i, j] = np.abs(order) * wave.to(u.mm).value / np.sin(theta_needed)
        e._d = RectBivariateSpline(pos_on_e[0][0, :], pos_on_e[1][:, 0], d_needed)

def chirp_flat_grating2(conf, gas, order, wave):
    '''
    Parameters
    ----------
    wave : `astropy.quantity.Quantity`
    '''
    focalpoint, ref_point = find_where_ref_ray_should_go(conf, order, wave)
    for e in gas.elements:
        d_needed = np.zeros(3)
        l_x = np.linalg.norm(e.geometry['v_y'])
        pos_on_e = np.array([-l_x, 0, l_x])
        for i in range(3):
            position = h2e(e.geometry['center']) + pos_on_e[i] * h2e(e.geometry['e_y'])
            vec_pos_foc = - position[:2] + h2e(focalpoint)[:2]
            vec_pos_foc = vec_pos_foc / np.linalg.norm(vec_pos_foc)
            vec_pos_ref_point = - position[:2] + h2e(ref_point)[:2]
            vec_pos_ref_point = vec_pos_ref_point / np.linalg.norm(vec_pos_ref_point)

            theta_needed = np.arccos(np.dot(vec_pos_foc, vec_pos_ref_point))
            d_needed[i] = np.abs(order) * wave.to(u.mm).value / np.sin(theta_needed)
        e._d_needed = d_needed
        e._chirp = (d_needed[2] - d_needed[0]) / (2 * l_x)
        def func(intercoos):
            return intercoos[:, 0] * e._chirp + d_needed[0]
        e._d = func
