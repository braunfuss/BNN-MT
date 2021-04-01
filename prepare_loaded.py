import ray
import numpy as np
from pyrocko import model, orthodrome, gf, util
from pyrocko.gf import LocalEngine, Target
from pyrocko import trace, io, cake
import copy
import psutil
import _pickle as pickle
from pathlib import Path
import scedc_util
import waveform_processing as wp
from mtqt_source import MTQTSource
num_cpus = psutil.cpu_count(logical=False)
pi = np.pi
earthradius_equator = 6378.14 * 1000.
d2m = earthradius_equator*pi/180.
m2d = 1./d2m


def assoicate_single(ev, data_dir, store_id, store,
                     stations=None, pre=0.5,
                     post=3, reference_event=None, min_len=420,
                     pick_sigma=0.02):
    events = []
    waveforms = []
    labels = []
    gf_freq = store.config.sample_rate
    mod = store.config.earthmodel_1d
    found = False
    pathlist = Path(data_dir).glob('ev_*/')
    for path in sorted(pathlist):
        targets = []
        path = str(path)+"/"
        try:
            event = model.load_events(path+"event.txt")[0]
            if ev.time-10 < event.time and ev.time+10 > event.time:
                traces_loaded = io.load(path+"/waveforms/rest/traces.mseed")
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
                traces = wp.check_traces(traces_loaded, stations, min_len=min_len)

                traces_processed, nsamples = wp.process_loaded_waveforms(traces,
                                                                         stations,
                                                                         ev,
                                                                         gf_freq,
                                                                         mod,
                                                                         pre,
                                                                         post)
                if found is False:
                    events.append(event)
                    waveforms.append(traces_processed)
                    found = True
        except:
            pass
    data_events, nsamples = wp.prepare_waveforms(waveforms)
    return data_events, nsamples, event


def assoicate(events_mts, data_dir, store_id, store,
              stations=None, pre=0.5,
              post=3, reference_event=None, min_len=420,
              pick_sigma=0.02):
    events = []
    waveforms = []
    labels = []
    gf_freq = store.config.sample_rate
    mod = store.config.earthmodel_1d

    for ev in events_mts:
        pathlist = Path(data_dir).glob('ev_*/')
        for path in sorted(pathlist):
            targets = []
            path = str(path)+"/"
            event = model.load_events(path+"event.txt")[0]
            if ev.time-10 < event.time and ev.time+10 > event.time:
                traces_loaded = io.load(path+"/waveforms/rest/traces.mseed")
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
                traces = wp.check_traces(traces_loaded, stations, min_len=min_len)
                traces_processed, nsamples = wp.process_loaded_waveforms(traces,
                                                                      stations,
                                                                      event,
                                                                      gf_freq,
                                                                      mod,
                                                                      pre, post)
                events.append(event)
                waveforms.append(traces_processed)

    data_events, nsamples = wp.prepare_waveforms(waveforms)
    return data_events, nsamples, event


def prep_data_batch(data_dir, store_id, stations=None, pre=0.5,
                    post=3, reference_event=None, min_len=420,
                    pick_sigma=0.02):
    engine = LocalEngine(store_superdirs=['/home/asteinbe/gf_stores'])
    store = engine.get_store(store_id)
    mod = store.config.earthmodel_1d
    gf_freq = store.config.sample_rate
    cake_phase = cake.PhaseDef("P")
    phase_list = [cake_phase]
    events = []
    waveforms = []
    waveforms_shifted = []
    events = scedc_util.scedc_fm_to_pyrocko(file)
    labels = labels_from_events(events)
    pathlist = Path(data_dir).glob('ev_0/')
    for path in sorted(pathlist):
        try:
            targets = []
            path = str(path)+"/"
            event = model.load_events(path+"event.txt")[0]
            traces_loaded = io.load(path+"traces.mseed")
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
            traces = check_traces(traces_loaded, stations, min_len=min_len)
            traces_processed, nsamples = wp.process_loaded_waveforms(traces,
                                                                  stations,
                                                                  event,
                                                                  gf_freq,
                                                                  mod,
                                                                  pre, post)
            events.append(event)
            waveforms.append(traces_processed)
        except:
            pass
    return waveforms, nsamples, events, waveforms_shifted
