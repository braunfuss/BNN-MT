import ray
import numpy as np
from pyrocko import model, orthodrome, gf, util
from pyrocko.gf import LocalEngine, Target
from pyrocko import trace, io, cake, model
import copy
import psutil
import _pickle as pickle
from pathlib import Path

from mtqt_source import MTQTSource
np.seterr(divide='ignore', invalid='ignore')
num_cpus = psutil.cpu_count(logical=False)
pi = np.pi
earthradius_equator = 6378.14 * 1000.
d2m = earthradius_equator*pi/180.
m2d = 1./d2m


@ray.remote
def get_parallel_mtqt(i, targets, store_id, post, pre, stations, mod, params,
                      strikes, dips, rakes, vs, ws,
                      store_dirs,
                      batch_loading=192, npm=0,
                      dump_full=False, seiger1f=False, path_count=0,
                      paths_disks=None, con_line=True, max_rho=0.,
                      mag=-6):

    engine = LocalEngine(store_superdirs=[store_dirs])
    store = engine.get_store(store_id)
    lat, lon, depth = params
    traces_uncuts = []
    tracess = []
    sources = []
    mtqt_ps = []
    count = 0
    npm = len(strikes)*len(dips)*len(rakes)*len(vs)*len(ws)
    npm_rem = npm
    data_events = []
    labels_events = []
    events = []
    if seiger1f is True:
        current_path = paths_disks[path_count]
    k = 0
    for strike in strikes:
        for dip in dips:
            for rake in rakes:
                for v in vs:
                    for w in ws:
                        name = "scenario"+str(i)
                        event = model.Event(name=name, lat=lat, lon=lon,
                                            magnitude=mag,
                                            depth=depth)
                        kappa = strike
                        sigma = rake
                        h = dip

                        source_mtqt = MTQTSource(
                            lon=lon,
                            lat=lat,
                            depth=depth,
                            w=w,
                            v=v,
                            kappa=kappa,
                            sigma=sigma,
                            h=h,
                            magnitude=mag
                        )

                        response = engine.process(source_mtqt, targets)
                        traces_synthetic = response.pyrocko_traces()

                        event.moment_tensor = source_mtqt.pyrocko_moment_tensor()

                        if dump_full is True:
                            traces_uncut = copy.deepcopy(traces)
                            traces_uncuts.append(traces_uncut)
                        traces = []
                        for tr in traces_synthetic:
                            for st in stations:
                                if st.station == tr.station:
                                    processed = False
                                    dist = (orthodrome.distance_accurate50m(source_mtqt.lat,
                                                                         source_mtqt.lon,
                                                                         st.lat,
                                                                         st.lon)+st.elevation)#*cake.m2d
                                    while processed is False:
                                        processed = False
                                        depth = source_mtqt.depth
                                        arrival = store.t('P', (depth, dist))

                                        if processed is False:
                                            tr.chop(arrival-pre, arrival+post)
                                            traces.append(tr)
                                            processed = True
                                        else:
                                            pass
                            nsamples = len(tr.ydata)

                        rho = 1
                        mtqt_ps = [[rho, v, w, kappa, sigma, h]]

                        data_event, nsamples = prepare_waveforms([traces])
                        label_event = prepare_labels([event], mtqt_ps)
                        data_events.append(data_event)
                        labels_events.append(label_event)
                        events.append(event)

                        if count == batch_loading or npm_rem < batch_loading:
                            npm_rem = npm_rem - batch_loading
                            k = k+1
                            if seiger1f is True:
                                free = os.statvfs(current_path)[0]*os.statvfs(current_path)[4]
                                if free < 80000:
                                    current_path = paths_disks[path_count+1]
                                    path_count = path_count + 1
                                f = open(current_path+"grid_%s_%s_%s_%s/batch_%s_grid_%s_SDR%s_%s_%s_%s_%s" % (store_id, lat, lon, int(depth), count, i, strike, dip, rake, v, w), 'ab')
                                pickle.dump([data_events, labels_events, nstations, nsamples, events], f)
                                f.close()
                            else:
                                util.ensuredir("grids/grid_%s_%s_%s_%s/" % (store_id, lat, lon, int(depth)))
                                f = open("grids/grid_%s_%s_%s_%s/batch_%s_grid_%s_SDR%s_%s_%s_%s_%s" % (store_id, lat, lon, int(depth), count, i, strike, dip, rake, v, w), 'ab')
                                pickle.dump([data_events, labels_events, nsamples, events], f)
                                f.close()

                            count = 0
                            data_events = []
                            labels_events = []
                            events = []
                            traces_uncuts = []
                            mtqt_ps = []

                        else:
                            count = count+1
    return []


def generate_test_data_grid(store_id, store_dirs, coordinates, geometry_params,
                            pre=0.5, post=3, stations_input=None,
                            batch_loading=256, paths_disks=None):

    engine = LocalEngine(store_superdirs=[store_dirs])
    store = engine.get_store(store_id)
    mod = store.config.earthmodel_1d
    cake_phase = cake.PhaseDef("P")
    phase_list = [cake_phase]

    waveforms_events = []
    waveforms_events_uncut = []
    waveforms_noise = []
    sources = []

    lats = coordinates[0]
    lons = coordinates[1]
    depths = coordinates[2]

    if stations_input is None:
        stations_unsorted = model.load_stations("data/stations.pf")
    else:
        stations_unsorted = model.load_stations(stations_input)
    for st in stations_unsorted:
        st.dist = orthodrome.distance_accurate50m(st.lat, st.lon, lats[0], lons[0])
        st.azi = orthodrome.azimuth(st.lat, st.lon, lats[0], lons[0])
    stations = sorted(stations_unsorted, key=lambda x: x.dist, reverse=True)

    targets = []
    events = []
    mean_lat = []
    mean_lon = []
    max_rho = 0.
    for st in stations:
        mean_lat.append(st.lat)
        mean_lon.append(st.lon)
        for cha in st.channels:
            if cha.name is not "R" and cha.name is not "T" and cha.name is not "Z":
                target = Target(
                        lat=st.lat,
                        lon=st.lon,
                        store_id=store_id,
                        interpolation='multilinear',
                        quantity='displacement',
                        codes=st.nsl() + (cha.name,))
                targets.append(target)

    strikes = geometry_params[0]
    dips = geometry_params[1]
    rakes = geometry_params[2]
    vs = geometry_params[3]
    ws = geometry_params[4]

    grid_points = []
    for lat in lats:
        for lon in lons:
            for depth in depths:
                grid_points.append([lat, lon, depth])

    ray.init(num_cpus=num_cpus-1)
    npm = len(lats)*len(lons)*len(depths)
    npm_geom = len(strikes)*len(dips)*len(rakes)

    results = ray.get([get_parallel_mtqt.remote(i, targets, store_id, post,
                                                pre, stations, mod,
                                                grid_points[i], strikes, dips,
                                                rakes, vs, ws, store_dirs,
                                                batch_loading=batch_loading,
                                                npm=npm_geom,
                                                paths_disks=paths_disks)
                      for i in range(len(grid_points))])
    ray.shutdown()
    return waveforms_events


def process_loaded_waveforms_shift(traces_in, stations, event, gf_freq, mod,
                                   pre, post, n_shift=2, shift_max=0.2,
                                   random_shift=False):

    traces_processed_shifted = []
    shifts = np.arange(-shift_max, shift_max, n_shift)
    traces_in.sort(key=lambda a: a.full_id)
    traces_sorted = trace.degapper(traces_in,
                                   maxgap=300000000,
                                   fillmethod='zeros',
                                   deoverlap='use_second',
                                   maxlap=None)
    nstations = len(stations)
    for ishift in range(0, n_shift):
        traces_processed = []
        traces = copy.deepcopy(traces_sorted)
        for st in stations:
            for tr in traces:
                nsamples = len(tr.ydata)
                if st.station == tr.station:
                    processed = False
                    dists = (orthodrome.distance_accurate50m(event.lat,
                                                             event.lon,
                                                             st.lat,
                                                             st.lon)+st.elevation)*cake.m2d
                    arrivals = mod.arrivals([dists],
                                            phases=["p", "P", "S", "s"],
                                            zstart=event.depth)
                    for i, arrival in enumerate(arrivals):
                        if processed is False:
                            try:
                                shift = shifts[i]
                                tr.chop(event.time+arrival.t-pre+shift,
                                        event.time+arrival.t+post+shift,
                                        want_incomplete=True)
                                if 1./gf_freq != tr.deltat:
                                    tr.resample(1./gf_freq)
                                traces_processed.append(tr)
                                processed = True
                            except:
                                data_zeros = np.zeros(int(500*(1/tr.deltat)))
                                t1 = trace.Trace(
                                    station=st.station, channel=tr.channel,
                                    deltat=tr.deltat, tmin=event.time+arrival.t-pre-5,
                                    ydata=data_zeros)
                                if np.isnan(np.max(tr.ydata)) is False:
                                    t1.add(tr)
                                    t1.chop(event.time+arrival.t-pre+shift,
                                            event.time+arrival.t+post+shift,
                                            want_incomplete=True)
                                    tr = t1
                                    if 1./gf_freq != tr.deltat:
                                        tr.resample(1./gf_freq)
                                    traces_processed.append(tr)
                                    processed = True
                                else:
                                    tr.ydata = np.nan_to_num(tr.ydata)
                                    t1.add(tr)
                                    t1.chop(event.time+arrival.t-pre+shift,
                                            event.time+arrival.t+post+shift,
                                            want_incomplete=True)
                                    tr = t1
                                    if 1./gf_freq != tr.deltat:
                                        tr.resample(1./gf_freq)
                                    traces_processed.append(tr)
                                    processed = True
        traces_processed_shifted.append(traces_processed)

    return traces_processed_shifted


def process_loaded_waveforms(traces_unsorted, stations, event, gf_freq, mod,
                             pre, post):
    traces_processed = []

    traces_unsorted.sort(key=lambda a: a.full_id)
    traces = trace.degapper(traces_unsorted,
                            maxgap=300000000,
                            fillmethod='zeros',
                            deoverlap='use_second',
                            maxlap=None)
    for st in stations:
        for tr in traces:
            nsamples = len(tr.ydata)
            if st.station == tr.station:
                dists = (orthodrome.distance_accurate50m(event.lat,
                                                         event.lon,
                                                         st.lat,
                                                         st.lon)+st.elevation)*cake.m2d
                processed = False
                for i, arrival in enumerate(mod.arrivals([dists],
                                            phases=["p", "P", "S"],
                                            zstart=event.depth)):
                    if processed is False:
                        try:
                            tr.chop(event.time+arrival.t-pre,
                                    event.time+arrival.t+post,
                                    want_incomplete=True)
                            if 1./gf_freq != tr.deltat:
                                tr.resample(1./gf_freq)
                            traces_processed.append(tr)
                            processed = True
                        except:
                            data_zeros = np.zeros(int(500*(1/tr.deltat)))
                            t1 = trace.Trace(
                                station=st.station, channel=tr.channel,
                                deltat=tr.deltat, tmin=event.time+arrival.t-pre-5,
                                ydata=data_zeros)
                            if np.isnan(np.max(tr.ydata)) is False:
                                t1.add(tr)
                                t1.chop(event.time+arrival.t-pre,
                                        event.time+arrival.t+post,
                                        want_incomplete=True)
                                tr = t1
                                if 1./gf_freq != tr.deltat:
                                    tr.resample(1./gf_freq)
                                traces_processed.append(tr)
                                processed = True

    return traces_processed, nsamples


def check_traces(traces_loaded, stations, fill_with_zeros=True, min_len=420,
                 event=None):
    traces = []
    for st in stations:
        found_E = False
        found_N = False
        found_Z = False

        for tr in traces_loaded:
            if st.station == tr.station:
                if len(tr.ydata) > min_len:
                    traces.append(tr)
                    if tr.channel[-1] == "E":
                        found_E = True
                    if tr.channel[-1] == "N":
                        found_N = True
                    if tr.channel[-1] == "Z":
                        found_Z = True
        if found_Z is False and fill_with_zeros is True:
            data_zeros = np.zeros(int(30*(1/tr.deltat)))
            t1 = trace.Trace(
                station=st.station, channel="Z",
                deltat=tr.deltat, tmin=event.time, ydata=data_zeros)
            traces.append(t1)
        if found_E is False and fill_with_zeros is True:
            data_zeros = np.zeros(int(30*(1/tr.deltat)))
            t1 = trace.Trace(
                station=st.station, channel="E",
                deltat=tr.deltat, tmin=event.time, ydata=data_zeros)
            traces.append(t1)
        if found_N is False and fill_with_zeros is True:
            data_zeros = np.zeros(int(30*(1/tr.deltat)))
            t1 = trace.Trace(
                station=st.station, channel="N",
                deltat=tr.deltat, tmin=event.time, ydata=data_zeros)
            traces.append(t1)
    return traces


def load_data(data_dir, store_id, stations=None, pre=0.5,
              post=3, reference_event=None, min_len=420, error_t=None,
              lat=None, lon=None, depth=None, engine=None,
              ev_id=None):
    store = engine.get_store(store_id)
    mod = store.config.earthmodel_1d
    gf_freq = store.config.sample_rate
    cake_phase = cake.PhaseDef("P")
    phase_list = [cake_phase]
    events = []
    waveforms = []
    waveforms_shifted = []
    if ev_id is None:
        pathlist = Path(data_dir).glob('ev_*/')
    else:
        pathlist = Path(data_dir).glob('ev_%s/' % ev_id)
    for path in sorted(pathlist):
        targets = []
        path = str(path)+"/"
        event = model.load_events(path+"event.txt")[0]
        traces_loaded = io.load(path+"waveforms/rest/traces.mseed")
        stations_unsorted = model.load_stations(data_dir+"stations.pf")
        for st in stations_unsorted:
            st.dist = orthodrome.distance_accurate50m(st.lat, st.lon,
                                                      event.lat,
                                                      event.lon)
            st.azi = orthodrome.azimuth(st.lat, st.lon, event.lat,
                                        event.lon)
        stations = sorted(stations_unsorted, key=lambda x: x.dist,
                          reverse=True)

        traces_processed = []
        if lat is not None:
            event.lat = lat
            event.lon = lon
            event.depth = depth

        traces = check_traces(traces_loaded, stations, min_len=min_len,
                              event=event)
        if error_t is not None:
            traces_shift = copy.deepcopy(traces)
        traces_processed, nsamples = process_loaded_waveforms(traces,
                                                              stations,
                                                              event,
                                                              gf_freq,
                                                              mod,
                                                              pre, post)
        if error_t is not None:

            traces_processed_shifted = process_loaded_waveforms_shift(traces_shift,
                                                                      stations,
                                                                      event,
                                                                      gf_freq,
                                                                      mod,
                                                                      pre, post,
                                                                      shift_max=error_t)
            waveforms_shifted.append(traces_processed_shifted)
        events.append(event)
        waveforms.append(traces_processed)

    return waveforms, nsamples, events, waveforms_shifted


def read_wavepickle(path):
    f = open(path, 'rb')
    data_events, labels_events, nsamples, events = pickle.load(f)
    f.close()
    return data_events, labels_events


def normalize_by_std_deviation(traces):
    traces_data = []
    for tr in traces:
        trace_level = np.nanmean(tr.ydata)
        tr.ydata = tr.ydata - trace_level
        traces_data.append(tr.ydata)
        tr.ydata = tr.ydata + trace_level

    traces_data = np.asarray(traces_data)
    nanstd = np.nanstd(traces_data, axis=1)[:, np.newaxis]
    nanstd[nanstd == 0] = 1.

    for i, tr in enumerate(traces):
        tr.ydata = tr.ydata/nanstd[i]
    return traces


def normalize_by_tracemax(traces):
    traces_data = []
    for tr in traces:
        trace_level = np.nanmean(tr.ydata)
        tr.ydata = tr.ydata - trace_level
        tr.ydata = tr.ydata/np.max(tr.ydata)
        tr.ydata = tr.ydata + trace_level

    return traces


def normalize(traces):
    traces_data = []
    for tr in traces:
        trace_level = np.nanmean(tr.ydata)
        tr.ydata = tr.ydata - trace_level
        tr.ydata = (tr.ydata-np.min(tr.ydata))/(np.max(tr.ydata)-np.min(tr.ydata))
    return traces


def normalize_chunk(traces):
    traces_data = []
    for tr in traces:
        traces_data.append(tr.ydata)

    traces_data = np.asarray(traces_data)
    max = np.max(traces_data)
    min = np.min(traces_data)

    for tr in traces:
        tr.ydata = (tr.ydata-min)/(max-min)
    return traces


def normalize_all(traces, min, max):

    for tr in traces:
        trace_level = np.nanmean(tr.ydata)
        tr.ydata = tr.ydata - trace_level
        tr.ydata = (tr.ydata-min)/(max-min)
    return traces


def prepare_labels(events, mtqt_ps):
    lons = []
    lats = []
    depths = []
    labels = []
    for i, ev in enumerate(events):
        mtqt_p = mtqt_ps[i]
        labels.append([
                       mtqt_p[3]/(2.*pi),
                       0.5-(mtqt_p[4]/(pi/2))*0.5,
                       mtqt_p[5],
                       0.5-(mtqt_p[1]/(1/3))*0.5,
                       0.5-(mtqt_p[2]/((3/8)*pi))*0.5])
    labels = np.asarray(labels)

    return labels


def prepare_waveforms(waveforms):
    data_traces = []
    maxsamples = 0
    for traces in waveforms:
        traces_event = []
        for tr in traces:
            tr.lowpass(4, 5.4)
            tr.highpass(4, 1.)
        traces_orig = copy.deepcopy(traces)
        traces_rel_max_values = np.zeros(len(traces))
        traces_rel_max_values_stations = []
        for i, tr_or in enumerate(traces_orig):
            for tr in traces_orig:
                if tr.station == tr_or.station:
                    if traces_rel_max_values[i] == 0:
                        traces_rel_max_values[i] = np.max(abs(tr.ydata))
                    else:
                        if np.max(abs(tr.ydata)) > traces_rel_max_values[i]:
                            traces_rel_max_values[i] = np.max(abs(tr.ydata))
            if len(tr.ydata) > maxsamples:
                maxsamples = len(tr.ydata)
        count_traces = 0
        for tr in traces:
            tr.ydata = 0.5 - (tr.ydata/traces_rel_max_values[count_traces])*0.5
            count_traces = count_traces + 1

            nsamples = len(tr.ydata)
            data = tr.ydata
            nsamples = len(data)
            if nsamples != maxsamples:
                data = np.pad(data, (0, maxsamples-nsamples), 'constant')
            inds = None
            inds = np.where(np.isnan(tr.ydata))
            # deal with empty traces
            if len(inds) != 0:
                for ind in inds:
                    tr.ydata[ind] = 0
            data = tr.ydata
            traces_event.append(data)
            nsamples = len(data)
        data_traces.append(traces_event)

    return data_traces, nsamples
