import _pickle as pickle
import numpy as np
from mtqt_source import MTQTSource
from pathlib import Path
from pyrocko import orthodrome
import tensorflow_probability as tfp
import waveform_processing as wp

tfd = tfp.distributions
pi = np.pi


def find_closest_grid_point(lat_ev, lon_ev, depth_ev, path_models=None,
                            gf_store_id=None, min_dist=7000.,
                            min_dist_depth=10000.):
    pathlist = Path(path_models).glob('model_%s_*' % gf_store_id)
    k = 0
    for path in sorted(pathlist):
        path = str(path)
        model_coordinates = path.split("_")
        lat = float(model_coordinates[3])
        lon = float(model_coordinates[4])
        depth = float(model_coordinates[5])
        dist = orthodrome.distance_accurate50m(lat_ev, lon_ev, lat, lon)
        if dist < min_dist:
            min_dist = dist
            dist_depth = abs(depth-depth_ev)
            if dist_depth < min_dist_depth:
                min_dist_depth = dist_depth
                best_model = path
        k = k+1
    return best_model


def grid_points_in_error_ellipses(lat_ev, lon_ev, depth_ev, error_h, error_z,
                                  path_models=None, gf_store_id=None):

    pathlist = Path(path_models).glob('model_%s_*' % gf_store_id)
    region = orthodrome.radius_to_region(lat_ev, lon_ev, error_h)
    grid_points = []
    for path in sorted(pathlist):
        path = str(path)
        model_coordinates = path.split("_")
        lat = float(model_coordinates[3])
        lon = float(model_coordinates[4])
        depth = float(model_coordinates[5])
        dists = orthodrome.distance_accurate50m_numpy(lat_ev, lon_ev, lat, lon)
        if dists < error_h:
            if depth_ev-error_z < depth and depth_ev+error_z > depth:
                grid_points.append(path)

    return grid_points


def find_event(path_events, time):
    pathlist = Path(path_events).glob('ev_*')
    for path in sorted(pathlist):
        path = str(path)+"/"
        event = model.load_events(path+"event.txt")[0]
        if time-10 < event.time and time+10 > event.time:
            return event, path


def loss_function_negative_log_likelihood():
    neg_log_likelihood = lambda x, rv_x: -rv_x.log_prob(x)
    return neg_log_likelihood


def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    c = np.log(np.expm1(1.))
    return tf.keras.Sequential([
      tfp.layers.VariableLayer(2 * n, dtype=dtype),
      tfp.layers.DistributionLambda(lambda t: tfd.Independent(
          tfd.Normal(loc=t[..., :n],
                     scale=1e-5 + tf.nn.softplus(c + t[..., n:])),
          reinterpreted_batch_ndims=1)),
    ])


# Specify the prior over `keras.layers.Dense` `kernel` and `bias`.
def prior_trainable(kernel_size, bias_size=0, dtype=None):
  n = kernel_size + bias_size
  return tf.keras.Sequential([
      tfp.layers.VariableLayer(n, dtype=dtype),
      tfp.layers.DistributionLambda(lambda t: tfd.Independent(
          tfd.Normal(loc=t, scale=1),
          reinterpreted_batch_ndims=1)),
  ])


def lambda_dist(scale=1e-3):
    return lambda t: tfd.Normal(loc=t, scale=scale)


def getitem__all_values(filenames, idx, batch_size=72):
    batch_x = filenames[idx]
    data = []
    labels = []
    for i in range(len(filenames)):
        batch_x = filenames[i]
        f = open(batch_x, 'rb')
        data_events, labels_events, nsamples,\
            events = pickle.load(f)
        f.close()
        for d, l in zip(data_events, labels_events):
            labels.append(l[0])
            d = d[0]
            d = np.asarray(d)
            d = d.reshape(d.shape+(1,))
            data.append(d)
    return np.array(data), np.array(labels), events


def waveform_2dGenerator_from_files(filenames, batchsize=72):
    batchsize = batchsize
    while 1:
        data = []
        labels = []
        for i in range(len(filenames)):
            batch_x = filenames[i]
            f = open(batch_x, 'rb')
            data_events, labels_events, nsamples,\
                events = pickle.load(f)
            f.close()
            for d, l in zip(data_events, labels_events):
                labels.append(l[0])
                d = d[0]
                d = np.asarray(d)
                d = d.reshape(d.shape+(1,))
                data.append(d)
                if len(labels) == batchsize:
                    yield np.array(data), np.array(labels)
                    data = []
                    labels = []


def convert_norm2real(values):
    true_mts = []
    true_values = []
    for p in values:
        p = p[0]
        v, w, kappa, sigma, h = p[3], p[4], p[0], p[1], p[2]

        v = (1/3)-(((1/3)*2)*v)
        w = ((3/8)*pi)-((((3/8)*pi)*2)*w)
        kappa = kappa*2.*pi
        sigma = (pi/2)-(2*(pi/2)*sigma)
        h = h
        if h > 1.:
            h = 1.
        if v > 1.:
            v = 1.
        mtqt_source = MTQTSource(v=v, w=w, kappa=kappa, sigma=sigma,
                                 h=h)
        mt = mtqt_source.pyrocko_moment_tensor()
        M = mtqt_source.m6
        true_mts.append(mt)
        true_values.append(M)
    return true_mts, true_values


def convert_waveforms_to_input(waveforms):
    waveforms_events = [waveforms[:]]
    data_events, nsamples = wp.prepare_waveforms(waveforms_events)
    data_events = np.asarray(data_events)
    data_events = data_events.reshape((data_events.shape[0],)+data_events.shape[1:]+(1,))
    data_events = np.float32(data_events)
    return data_events


def convert_data_events_to_input(data_events):
    data_events = np.asarray(data_events)
    data_events = data_events.reshape((data_events.shape[0],)+data_events.shape[1:]+(1,))
    data_events = np.float32(data_events)
    return data_events


def getitem_values(filenames, batch_size, idx):
    batch_x = filenames[idx]
    f = open(batch_x, 'rb')
    data_events, labels_events, nsamples, events = pickle.load(f)
    f.close()
    return np.array(data_events), np.array(labels_events), events
