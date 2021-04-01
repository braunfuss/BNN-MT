from pyrocko.gf import meta
from pyrocko.gf.seismosizer import Source
from pyrocko import gf
from pyrocko.guts import Float
from pyrocko import moment_tensor as mtm

import numpy as num
pi = num.pi
pi4 = pi / 4.
km = 1000.
d2r = pi / 180.
r2d = 180. / pi

sqrt3 = num.sqrt(3.)
sqrt2 = num.sqrt(2.)
sqrt6 = num.sqrt(6.)


class MTQTSource(gf.SourceWithMagnitude):
    """
    A moment tensor point source.
    Notes
    -----
    Following Q-T parameterization after Tape & Tape 2015
    """

    discretized_source_class = meta.DiscretizedMTSource

    w = Float.T(
        default=0.,
        help='Lune latitude delta transformed to grid. '
             'Defined: -3/8pi <= w <=3/8pi. '
             'If fixed to zero the MT is deviatoric.')

    v = Float.T(
        default=0.,
        help='Lune co-longitude transformed to grid. '
             'Definded: -1/3 <= v <= 1/3. '
             'If fixed to zero together with w the MT is pure DC.')

    kappa = Float.T(
        default=0.,
        help='Strike angle equivalent of moment tensor plane.'
             'Defined: 0 <= kappa <= 2pi')

    sigma = Float.T(
        default=0.,
        help='Rake angle equivalent of moment tensor slip angle.'
             'Defined: -pi/2 <= sigma <= pi/2')

    h = Float.T(
        default=0.,
        help='Dip angle equivalent of moment tensor plane.'
             'Defined: 0 <= h <= 1')

    def __init__(self, **kwargs):
        n = 1000
        self._beta_mapping = num.linspace(0, pi, n)
        self._u_mapping = \
            (3. / 4. * self._beta_mapping) - \
            (1. / 2. * num.sin(2. * self._beta_mapping)) + \
            (1. / 16. * num.sin(4. * self._beta_mapping))

        self.lambda_factor_matrix = num.array(
            [[sqrt3, -1., sqrt2],
             [0., 2., sqrt2],
             [-sqrt3, -1., sqrt2]], dtype='float64')

        self.R = get_rotation_matrix()
        self.roty_pi4 = self.R['y'](-pi4)
        self.rotx_pi = self.R['x'](pi)

        self._lune_lambda_matrix = num.zeros((3, 3), dtype='float64')

        Source.__init__(self, **kwargs)

    @property
    def u(self):
        """
        Lunar co-latitude(beta), dependend on w
        """
        return (3. / 8.) * num.pi - self.w

    @property
    def gamma(self):
        """
        Lunar longitude, dependend on v
        """
        return v_to_gamma(self.v)

    @property
    def beta(self):
        """
        Lunar co-latitude, dependend on u
        """
        return w_to_beta(
            self.w, u_mapping=self._u_mapping, beta_mapping=self._beta_mapping)

    def delta(self):
        """
        From Tape & Tape 2012, delta measures departure of MT being DC
        Delta = Gamma = 0 yields pure DC
        """
        return (pi / 2.) - self.beta

    @property
    def rho(self):
        return mtm.magnitude_to_moment(self.magnitude) * sqrt2

    @property
    def theta(self):
        return num.arccos(self.h)

    @property
    def rot_theta(self):
        return self.R['x'](self.theta)

    @property
    def rot_kappa(self):
        return self.R['z'](-self.kappa)

    @property
    def rot_sigma(self):
        return self.R['z'](self.sigma)

    @property
    def lune_lambda(self):
        sin_beta = num.sin(self.beta)
        cos_beta = num.cos(self.beta)
        sin_gamma = num.sin(self.gamma)
        cos_gamma = num.cos(self.gamma)
        vec = num.array([sin_beta * cos_gamma, sin_beta * sin_gamma, cos_beta])
        return 1. / sqrt6 * self.lambda_factor_matrix.dot(vec) * self.rho

    @property
    def lune_lambda_matrix(self):
        num.fill_diagonal(self._lune_lambda_matrix, self.lune_lambda)
        return self._lune_lambda_matrix

    @property
    def rot_V(self):
        return self.rot_kappa.dot(self.rot_theta).dot(self.rot_sigma)

    @property
    def rot_U(self):
        return self.rot_V.dot(self.roty_pi4)

    @property
    def m9_nwu(self):
        """
        MT orientation is in NWU
        """
        return self.rot_U.dot(
            self.lune_lambda_matrix).dot(num.linalg.inv(self.rot_U))

    @property
    def m9(self):
        """
        Pyrocko MT in NED
        """
        return self.rotx_pi.dot(self.m9_nwu).dot(self.rotx_pi.T)

    @property
    def m6(self):
        return mtm.to6(self.m9)

    @property
    def m6_astuple(self):
        return tuple(self.m6.ravel().tolist())

    def base_key(self):
        return Source.base_key(self) + self.m6_astuple

    def discretize_basesource(self, store, target=None):
        times, amplitudes = self.effective_stf_pre().discretize_t(
            store.config.deltat, self.time)
        return meta.DiscretizedMTSource(
            m6s=self.m6[num.newaxis, :] * amplitudes[:, num.newaxis],
            **self._dparams_base_repeated(times))

    def pyrocko_moment_tensor(self):
        return mtm.MomentTensor(m=mtm.symmat6(*self.m6_astuple) * self.moment)

    def pyrocko_event(self, **kwargs):
        mt = self.pyrocko_moment_tensor()
        return Source.pyrocko_event(
            self,
            moment_tensor=self.pyrocko_moment_tensor(),
            magnitude=float(mt.moment_magnitude()),
            **kwargs)

    @classmethod
    def from_pyrocko_event(cls, ev, **kwargs):
        d = {}
        mt = ev.moment_tensor
        if mt:
            logger.warning(
                'From event will ignore MT components initially. '
                'Needs mapping from NED to QT space!')
            # d.update(m6=list(map(float, mt.m6())))

        d.update(kwargs)
        return super(MTQTSource, cls).from_pyrocko_event(ev, **d)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['R'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.R = get_rotation_matrix()


def vonmises_fisher(lats, lons, lats0, lons0, sigma=1.):
    """
    Von-Mises Fisher distribution function.
    Parameters
    ----------
    lats : float or array_like
        Spherical-polar latitude [deg][-pi/2 pi/2] to evaluate function at.
    lons : float or array_like
        Spherical-polar longitude [deg][-pi pi] to evaluate function at
    lats0 : float or array_like
        latitude [deg] at the center of the distribution (estimated values)
    lons0 : float or array_like
        longitude [deg] at the center of the distribution (estimated values)
    sigma : float
        Width of the distribution.
    Returns
    -------
    float or array_like
        log-probability of the VonMises-Fisher distribution.
    Notes
    -----
    Wikipedia:
        https://en.wikipedia.org/wiki/Von_Mises-Fisher_distribution
        modified from: https://github.com/williamjameshandley/spherical_kde
    """

    def logsinh(x):
        """ Compute log(sinh(x)), stably for large x.<
        Parameters
        ----------
        x : float or numpy.array
            argument to evaluate at, must be positive
        Returns
        -------
        float or numpy.array
            log(sinh(x))
        """
        if num.any(x < 0):
            raise ValueError("logsinh only valid for positive arguments")
        return x + num.log(1. - num.exp(-2. * x)) - num.log(2.)

    # transform to [0-pi, 0-2pi]
    lats_t = 90. + lats
    lons_t = 180. + lons
    lats0_t = 90. + lats0
    lons0_t = 180. + lons0

    x = cartesian_from_polar(
        phi=num.deg2rad(lons_t), theta=num.deg2rad(lats_t))
    x0 = cartesian_from_polar(
        phi=num.deg2rad(lons0_t), theta=num.deg2rad(lats0_t))

    norm = -num.log(4. * num.pi * sigma ** 2) - logsinh(1. / sigma ** 2)
    return norm + num.tensordot(x, x0, axes=[[0], [0]]) / sigma ** 2


def vonmises_std(lons, lats):
    """
    Von-Mises sample standard deviation.
    Parameters
    ----------
    phi, theta : array-like
        Spherical-polar coordinate samples to compute mean from.
    Returns
    -------
        solution for
        ..math:: 1/tanh(x) - 1/x = R,
        where
        ..math:: R = || sum_i^N x_i || / N
    Notes
    -----
    Wikipedia:
        https://en.wikipedia.org/wiki/Von_Mises-Fisher_distribution#Estimation_of_parameters
        but re-parameterised for sigma rather than kappa.
    modidied from: https://github.com/williamjameshandley/spherical_kde
    """
    from scipy.optimize import brentq

    x = cartesian_from_polar(phi=num.deg2rad(lons), theta=num.deg2rad(lats))
    S = num.sum(x, axis=-1)

    R = S.dot(S) ** 0.5 / x.shape[-1]

    def f(s):
        return 1. / num.tanh(s) - 1. / s - R

    kappa = brentq(f, 1e-8, 1e8)
    sigma = kappa ** -0.5
    return sigma


def cartesian_from_polar(phi, theta):
    """
    Embedded 3D unit vector from spherical polar coordinates.
    Parameters
    ----------
    phi, theta : float or numpy.array
        azimuthal and polar angle in radians.
        (phi-longitude, theta-latitude)
    Returns
    -------
    nhat : numpy.array
        unit vector(s) in direction (phi, theta).
    """
    x = num.sin(theta) * num.cos(phi)
    y = num.sin(theta) * num.sin(phi)
    z = num.cos(theta)
    return num.array([x, y, z])


def cartesian_from_polar(phi, theta):
    """
    Embedded 3D unit vector from spherical polar coordinates.
    Parameters
    ----------
    phi, theta : float or numpy.array
        azimuthal and polar angle in radians.
        (phi-longitude, theta-latitude)
    Returns
    -------
    nhat : numpy.array
        unit vector(s) in direction (phi, theta).
    """
    x = num.sin(theta) * num.cos(phi)
    y = num.sin(theta) * num.sin(phi)
    z = num.cos(theta)
    return num.array([x, y, z])


def kde2plot(x, y, grid=200, ax=None, **kwargs):
    if ax is None:
        _, ax = plt.subplots(1, 1, squeeze=True)
    kde2plot_op(ax, x, y, grid, **kwargs)
    return ax


def spherical_kde_op(
        lats0, lons0, lats=None, lons=None, grid_size=(200, 200), sigma=None):

    from scipy.special import logsumexp

    if sigma is None:

        sigmahat = vonmises_std(lats=lats0, lons=lons0)
        sigma = 1.06 * sigmahat * lats0.size ** -0.2

    if lats is None and lons is None:
        lats_vec = num.linspace(-90., 90, grid_size[0])
        lons_vec = num.linspace(-180., 180, grid_size[1])

        lons, lats = num.meshgrid(lons_vec, lats_vec)

    if lats is not None:
        assert lats.size == lons.size

    vmf = vonmises_fisher(
        lats=lats, lons=lons,
        lats0=lats0, lons0=lons0, sigma=sigma)
    kde = num.exp(logsumexp(vmf, axis=-1)).reshape(   # , b=self.weights)
        (grid_size[0], grid_size[1]))
    return kde, lats, lons


def v_to_gamma(v):
    """
    Converts from v parameter (Tape2015) to lune longitude [rad]
    """
    return (1. / 3.) * num.arcsin(3. * v)


def w_to_beta(w, u_mapping=None, beta_mapping=None, n=1000):
    """
    Converts from  parameter w (Tape2015) to lune co-latitude
    """
    if beta_mapping is None:
        beta_mapping = num.linspace(0, pi, n)

    if u_mapping is None:
        u_mapping = (
            3. / 4. * beta_mapping) - (
            1. / 2. * num.sin(2. * beta_mapping)) + (
            1. / 16. * num.sin(4. * beta_mapping))
    return num.interp(3. * pi / 8. - w, u_mapping, beta_mapping)


def w_to_delta(w, n=1000):
    """
    Converts from parameter w (Tape2015) to lune latitude
    """
    beta = w_to_beta(w)
    return pi / 2. - beta


def get_gmt_config(gmtpy, h=20., w=20.):

    if gmtpy.is_gmt5(version='newest'):
        gmtconfig = {
            'MAP_GRID_PEN_PRIMARY': '0.1p',
            'MAP_GRID_PEN_SECONDARY': '0.1p',
            'MAP_FRAME_TYPE': 'fancy',
            'FONT_ANNOT_PRIMARY': '14p,Helvetica,black',
            'FONT_ANNOT_SECONDARY': '14p,Helvetica,black',
            'FONT_LABEL': '14p,Helvetica,black',
            'FORMAT_GEO_MAP': 'D',
            'GMT_TRIANGULATE': 'Watson',
            'PS_MEDIA': 'Custom_%ix%i' % (w * gmtpy.cm, h * gmtpy.cm),
        }
    else:
        gmtconfig = {
            'MAP_FRAME_TYPE': 'fancy',
            'GRID_PEN_PRIMARY': '0.01p',
            'ANNOT_FONT_PRIMARY': '1',
            'ANNOT_FONT_SIZE_PRIMARY': '12p',
            'PLOT_DEGREE_FORMAT': 'D',
            'GRID_PEN_SECONDARY': '0.01p',
            'FONT_LABEL': '14p,Helvetica,black',
            'PS_MEDIA': 'Custom_%ix%i' % (w * gmtpy.cm, h * gmtpy.cm),
        }
    return gmtconfig


def lune_plot(v_tape=None, w_tape=None):

    from pyrocko import gmtpy

    if len(gmtpy.detect_gmt_installations()) < 1:
        raise gmtpy.GmtPyError(
            'GMT needs to be installed for lune_plot!')

    fontsize = 14
    font = '1'

    def draw_lune_arcs(gmt, R, J):

        lons = [30., -30., 30., -30.]
        lats = [54.7356, 35.2644, -35.2644, -54.7356]

        gmt.psxy(
            in_columns=(lons, lats), N=True, W='1p,black', R=R, J=J)

    def draw_lune_points(gmt, R, J, labels=True):

        lons = [0., -30., -30., -30., 0., 30., 30., 30., 0.]
        lats = [-90., -54.7356, 0., 35.2644, 90., 54.7356, 0., -35.2644, 0.]
        annotations = [
            '-ISO', '', '+CLVD', '+LVD', '+ISO', '', '-CLVD', '-LVD', 'DC']
        alignments = ['TC', 'TC', 'RM', 'RM', 'BC', 'BC', 'LM', 'LM', 'TC']

        gmt.psxy(in_columns=(lons, lats), N=True, S='p6p', W='1p,0', R=R, J=J)

        rows = []
        if labels:
            farg = ['-F+f+j']
            for lon, lat, text, align in zip(
                    lons, lats, annotations, alignments):

                rows.append((
                    lon, lat,
                    '%i,%s,%s' % (fontsize, font, 'black'),
                    align, text))

            gmt.pstext(
                in_rows=rows,
                N=True, R=R, J=J, D='j5p', *farg)

    def draw_lune_kde(
            gmt, v_tape, w_tape, grid_size=(200, 200), R=None, J=None):

        def check_fixed(a, varname):
            if a.std() == 0:
                a += num.random.normal(loc=0., scale=0.25, size=a.size)

        gamma = num.rad2deg(v_to_gamma(v_tape))   # lune longitude [rad]
        delta = num.rad2deg(w_to_delta(w_tape))   # lune latitude [rad]

        check_fixed(gamma, varname='v')
        check_fixed(delta, varname='w')

        lats_vec, lats_inc = num.linspace(
            -90., 90., grid_size[0], retstep=True)
        lons_vec, lons_inc = num.linspace(
            -30., 30., grid_size[1], retstep=True)
        lons, lats = num.meshgrid(lons_vec, lats_vec)

        kde_vals, _, _ = spherical_kde_op(
            lats0=delta, lons0=gamma,
            lons=lons, lats=lats, grid_size=grid_size)
        Tmin = num.min([0., kde_vals.min()])
        Tmax = num.max([0., kde_vals.max()])

        cptfilepath = '/tmp/tempfile.cpt'
        gmt.makecpt(
            C='white,yellow,orange,red,magenta,violet',
            Z=True, D=True,
            T='%f/%f' % (Tmin, Tmax),
            out_filename=cptfilepath, suppress_defaults=True)

        grdfile = gmt.tempfilename()
        gmt.xyz2grd(
            G=grdfile, R=R, I='%f/%f' % (lons_inc, lats_inc),
            in_columns=(lons.ravel(), lats.ravel(), kde_vals.ravel()),  # noqa
            out_discard=True)

        gmt.grdimage(grdfile, R=R, J=J, C=cptfilepath)

    h = 20.
    w = h / 1.9

    gmtconfig = get_gmt_config(gmtpy, h=h, w=w)
    bin_width = 15  # tick increment

    J = 'H0/%f' % (w - 5.)
    R = '-30/30/-90/90'
    B = 'f%ig%i/f%ig%i' % (bin_width, bin_width, bin_width, bin_width)

    gmt = gmtpy.GMT(config=gmtconfig)

    draw_lune_kde(
        gmt, v_tape=v_tape, w_tape=w_tape, grid_size=(300, 300), R=R, J=J)
    gmt.psbasemap(R=R, J=J, B=B)
    draw_lune_arcs(gmt, R=R, J=J)
    draw_lune_points(gmt, R=R, J=J)
    return gmt


def get_rotation_matrix(axes=['x', 'y', 'z']):
    """
    Return a function for 3-d rotation matrix for a specified axis.
    Parameters
    ----------
    axes : str or list of str
        x, y or z for the axis
    Returns
    -------
    func that takes an angle [rad]
    """
    ax_avail = ['x', 'y', 'z']
    for ax in axes:
        if ax not in ax_avail:
            raise TypeError(
                'Rotation axis %s not supported!'
                ' Available axes: %s' % (ax, list2string(ax_avail)))

    def rotx(angle):
        cos_angle = num.cos(angle)
        sin_angle = num.sin(angle)
        return num.array(
            [[1, 0, 0],
             [0, cos_angle, -sin_angle],
             [0, sin_angle, cos_angle]], dtype='float64')

    def roty(angle):
        cos_angle = num.cos(angle)
        sin_angle = num.sin(angle)
        return num.array(
            [[cos_angle, 0, sin_angle],
             [0, 1, 0],
             [-sin_angle, 0, cos_angle]], dtype='float64')

    def rotz(angle):
        cos_angle = num.cos(angle)
        sin_angle = num.sin(angle)
        return num.array(
            [[cos_angle, -sin_angle, 0],
             [sin_angle, cos_angle, 0],
             [0, 0, 1]], dtype='float64')

    R = {'x': rotx,
         'y': roty,
         'z': rotz}

    if isinstance(axes, list):
        return R
    elif isinstance(axes, str):
        return R[axes]
    else:
        raise Exception('axis has to be either string or list of strings!')
