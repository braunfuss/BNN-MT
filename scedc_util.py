import numpy as np
from pyrocko import model, util
from pyrocko import moment_tensor as pmt
pi = np.pi


def labels_from_events(events):
    labels = []
    for event in events:
        mt = event.moment_tensor
        strike = mt.strike1
        dip = mt.dip1
        rake = mt.rake1
        strike = (strike/360.)*2*pi
        dip = 1-(dip/90.)
        rake = (((rake-180)/180.))*(pi/2.)
        v = 0
        w = 0
        label = [strike, rake, dip, v, w]
        labels.append(label)
    return labels


def scedc_mt_to_pyrocko(moment_tensors):
    '''
    Need mtpar from https://github.com/rmodrak/mtpar/ as dependency.
    This converts moment tensors from the SCEDC website to labels to compare
    with estimates from the Machine Learning algorithm.
    '''
    from mtpar import cmt2tt15 # needed for conversion
    labels = []
    for mt in moment_tensors:
        mt_use = mt.m6_up_south_east()
        mt_input = []
        for mt_comp in mt_use:
            if mt_comp == 0:
                mt_comp += 1e-32
            else:
                mt_comp = mt_comp/mt.moment
            mt_input.append(mt_comp)
        rho, v, w, kappa, sigma, h = cmt2tt15(np.array(mt_input))
        strike = mt.strike1
        dip = mt.dip1
        rake = mt.rake1
        strike = (strike/360.)*2*pi
        dip = 1-(dip/90.)
        rake = (((rake-180)/180.))*(pi/2.)
        labels.append([strike, rake, dip, v, w])
    labels = np.asarray(labels)
    return labels


def get_all_scn_mechs():
    mechs = np.loadtxt("ridgecrest/scn_plot.mech", dtype="str")
    dates = []
    strikes = []
    rakes = []
    dips = []
    depths = []
    lats = []
    lons = []
    events = []
    for i in mechs:
        dates.append(i[1][0:4]+"-"+i[1][5:7]+"-"+i[1][8:]+" "+i[2])
        strikes.append(float(i[16]))
        dips.append(float(i[17]))
        rakes.append(float(i[18]))
        lats.append(float(i[7]))
        lons.append(float(i[8]))
        depths.append(float(i[9]))
        mt = pmt.MomentTensor(strike=float(i[16]), dip=float(i[17]), rake=float(i[18]),
                                        magnitude=float(i[5]))
        event = model.event.Event(lat=float(i[7]), lon=float(i[8]), depth=float(i[9]),
                                  moment_tensor=mt, magnitude=float(i[5]),
                                  time=util.str_to_time(i[1][0:4]+"-"+i[1][5:7]+"-"+i[1][8:]+" "+i[2]))
        events.append(event)
    return events


def scedc_fm_to_pyrocko(file):
    mechs = np.loadtxt(file, dtype="str")
    dates = []
    strikes = []
    rakes = []
    dips = []
    depths = []
    lats = []
    lons = []
    events = []
    errors_h = []
    errors_t = []
    errors_z = []
    for i in mechs:
        dates.append(i[1][0:4]+"-"+i[1][5:7]+"-"+i[1][8:]+" "+i[2])
        strikes.append(float(i[16]))
        dips.append(float(i[17]))
        rakes.append(float(i[18]))
        lats.append(float(i[7]))
        lons.append(float(i[8]))
        depths.append(float(i[9]))
        if float(i[13])*1000. > 4000.:
            errors_h.append(2000.)
        if float(i[13])*1000. < 1000.:
            errors_h.append(2000.)
        else:
            errors_h.append(6000.)
        if float(i[14])*1000. > 2000.:
            errors_z.append(2000.)
        else:
            errors_z.append(2000.)

        errors_t.append(float(i[15]))
        mt = pmt.MomentTensor(strike=float(i[16]), dip=float(i[17]),
                              rake=float(i[18]),
                              magnitude=float(i[5]))
        event = model.event.Event(lat=float(i[7]), lon=float(i[8]),
                                  depth=float(i[9]),
                                  moment_tensor=mt, magnitude=float(i[5]),
                                  time=util.str_to_time(i[1][0:4]+"-"+i[1][5:7]+"-"+i[1][8:]+" "+i[2]))
        events.append(event)
    return events, errors_h, errors_t, errors_z


def mxy2mt(M):
    mt = pmt.MomentTensor(mnn=M[0], mee=M[1], mdd=M[2], mne=M[3],
                          mnd=M[4], med=M[5])
    return mt


def mts_paper_scedc():
    '''
    Hardcoded copy of the moment tensors from the scedc website.
    Retrieved: 23.12.2020
    '''
    mts = []
    times = []
    events = []

    #https://service.scedc.caltech.edu/MomentTensor/solutions/web_38460967/ci38460967_MT.html
    time="2019-07-06 09:28:29"
    Mo=2.69e+23
    Mxx=-1.803e+23
    Mxy=-1.840e+23
    Mxz=2.476e+22
    Myy=2.068e+23
    Myz=-1.784e+22
    Mzz=-2.650e+22
    lat = 35.8978
    lon = -117.7250
    depth = 3900
    M = [Mxx, Myy, Mzz, Mxy, Mxz, Myz]
    mt = pmt.MomentTensor(mnn=M[0], mee=M[1], mdd=M[2], mne=M[3], mnd=M[4], med=M[5])
    mts.append(mt)
    times.append(time)
    event = model.Event(time=util.stt(time), moment_tensor=M, lat=lat,
                        lon=lon, depth=depth)
    events.append(event)

    # https://service.scedc.caltech.edu/MomentTensor/solutions/web_38463551/ci38463551_MT.html
    time="2019-07-06 13:06:55"
    Mo=6.91e+22
    Mxx=-4.574e+22
    Mxy=4.274e+21
    Mxz=1.557e+22
    Myy=7.768e+22
    Myz=-1.884e+22
    Mzz=-3.194e+22
    lat = 35.9280
    lon = -117.7063
    depth = 1500
    M = [Mxx, Myy, Mzz, Mxy, Mxz, Myz]
    mt = pmt.MomentTensor(mnn=M[0], mee=M[1], mdd=M[2], mne=M[3], mnd=M[4], med=M[5])
    mts.append(mt)
    times.append(time)
    event = model.Event(time=util.stt(time), moment_tensor=M, lat=lat,
                        lon=lon, depth=depth)
    events.append(event)

    # https://service.scedc.caltech.edu/MomentTensor/solutions/web_38466495/ci38466495_MT.html
    time="2019-07-06 17:59:15"
    Mo=8.70e+21
    Mxx=-4.447e+21
    Mxy=-6.931e+21
    Mxz=-1.326e+21
    Myy=5.598e+21
    Myz=9.049e+20
    Mzz=-1.151e+21
    lat = 35.8997
    lon = -117.7347
    depth = 3000
    M = [Mxx, Myy, Mzz, Mxy, Mxz, Myz]
    mt = pmt.MomentTensor(mnn=M[0], mee=M[1], mdd=M[2], mne=M[3], mnd=M[4], med=M[5])
    mts.append(mt)
    times.append(time)
    event = model.Event(time=util.stt(time), moment_tensor=M, lat=lat,
                        lon=lon, depth=depth)
    events.append(event)

    # https://service.scedc.caltech.edu/MomentTensor/solutions/web_38517119/ci38517119_MT.html
    time="2019-07-10 12:00:05"
    Mo=6.52e+21
    Mxx=-1.755e+21
    Mxy=-1.499e+21
    Mxz=3.140e+21
    Myy=6.118e+21
    Myz=-8.072e+20
    Mzz=-4.363e+21
    lat = 35.8765
    lon = -117.7068
    depth = 4400
    M = [Mxx, Myy, Mzz, Mxy, Mxz, Myz]
    mt = pmt.MomentTensor(mnn=M[0], mee=M[1], mdd=M[2], mne=M[3], mnd=M[4], med=M[5])
    mts.append(mt)
    times.append(time)
    event = model.Event(time=util.stt(time), moment_tensor=M, lat=lat,
                        lon=lon, depth=depth)
    events.append(event)

    # https://service.scedc.caltech.edu/MomentTensor/solutions/web_38538991/ci38538991_MT.html
    time="2019-07-11 23:45:19"
    Mo=1.97e+22
    Mxx=-8.999e+21
    Mxy=-1.598e+22
    Mxz=6.197e+21
    Myy=9.964e+21
    Myz=2.233e+21
    Mzz=-9.657e+20
    lat = 35.9482
    lon = -117.7057
    depth = 1500
    M = [Mxx, Myy, Mzz, Mxy, Mxz, Myz]
    mt = pmt.MomentTensor(mnn=M[0], mee=M[1], mdd=M[2], mne=M[3], mnd=M[4], med=M[5])
    mts.append(mt)
    times.append(time)
    event = model.Event(time=util.stt(time), moment_tensor=M, lat=lat,
                        lon=lon, depth=depth)
    events.append(event)

    #https://service.scedc.caltech.edu/MomentTensor/solutions/web_38644943/ci38644943_MT.html
    time="2019-07-26 00:42:48"
    Mo=1.62e+23
    Mxx=-6.930e+22
    Mxy=9.560e+21
    Mxz=3.263e+22
    Myy=1.860e+23
    Myz=-2.825e+22
    Mzz=-1.167e+23
    lat = 35.9237
    lon = -117.7115
    depth = 1900
    M = [Mxx, Myy, Mzz, Mxy, Mxz, Myz]
    mt = pmt.MomentTensor(mnn=M[0], mee=M[1], mdd=M[2], mne=M[3], mnd=M[4], med=M[5])
    mts.append(mt)
    times.append(time)
    event = model.Event(time=util.stt(time), moment_tensor=M, lat=lat,
                        lon=lon, depth=depth)
    events.append(event)

    # https://service.scedc.caltech.edu/MomentTensor/solutions/web_38996632/ci38996632_MT.html
    time="2019-08-22 20:49:50"
    Mo=2.72e+23
    Mxx=-8.042e+22
    Mxy=-2.026e+23
    Mxz=1.983e+22
    Myy=2.492e+23
    Myz=1.635e+22
    Mzz=-1.688e+23
    lat = 35.9077
    lon = -117.7092
    depth = 4900
    M = [Mxx, Myy, Mzz, Mxy, Mxz, Myz]
    mt = pmt.MomentTensor(mnn=M[0], mee=M[1], mdd=M[2], mne=M[3], mnd=M[4], med=M[5])
    mts.append(mt)
    times.append(time)
    event = model.Event(time=util.stt(time), moment_tensor=M, lat=lat,
                        lon=lon, depth=depth)
    events.append(event)

    # https://service.scedc.caltech.edu/MomentTensor/solutions/web_38999296/ci38999296_MT.html
    time="2019-08-23 05:34:10"
    Mo=3.87e+22
    Mxx=-1.383e+22
    Mxy=-2.949e+22
    Mxz=1.231e+22
    Myy=2.220e+22
    Myz=9.991e+21
    Mzz=-8.375e+21
    lat = 35.9078
    lon = -117.7047
    depth = 7200
    M = [Mxx, Myy, Mzz, Mxy, Mxz, Myz]
    mt = pmt.MomentTensor(mnn=M[0], mee=M[1], mdd=M[2], mne=M[3], mnd=M[4], med=M[5])
    mts.append(mt)
    times.append(time)
    event = model.Event(time=util.stt(time), moment_tensor=M, lat=lat,
                        lon=lon, depth=depth)
    events.append(event)

    return mts, times, events
