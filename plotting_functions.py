import matplotlib.pyplot as plt
from pyrocko import plot
from pyrocko.plot import beachball
from omega_angle import omega_angle
from pyrocko import moment_tensor as mtm
import numpy as np
import matplotlib
from matplotlib.gridspec import GridSpec
try:
    import seaborn as sns
except:
    pass

matplotlib.rc('xtick', labelsize=20)


def plot_prescission(input, output):
    mislocation_rel = []
    for inp, outp in zip(input, output):
        mislocation_rel.append(inp-outp)
    mislocation_rel = np.asarray(mislocation_rel)
    plt.figure()
    plt.plot(mislocation_rel)
    plt.show()


def plot_map_basemap(stations=None, east_min=-119.2, east_max=-116, north_min=34.5,
                     north_max=37.501, events=None, savename=None, preds=None,
                     best_mts=None, pred_events=None,
                     rect_lats=None,
                     rect_lons=None, ticks=0.01, kmscale=5, add_grid=True,
                     overview=False):
    try:
        from mpl_toolkits.basemap import Basemap
        use_basemap = True
    except:
        import cartopy.crs as ccrs
        import cartopy
        import cartopy.geodesic as cgeo
        from cartopy.io import srtm
        from cartopy.io import PostprocessedRasterSource, LocatedImage
        from cartopy.io.srtm import SRTM3Source, SRTM1Source
        use_basemap = False
    from matplotlib import pyplot as plt
    from pyrocko.plot import beachball
    from pyrocko import plot
    from obspy.imaging.beachball import beach
    fig, ax = plt.subplots(figsize=(12,12))
    map = Basemap(projection='merc', llcrnrlon=east_min,
                  llcrnrlat=north_min,urcrnrlon=east_max,
                  urcrnrlat=north_max,
                  resolution='h', epsg=3395, ax=ax)

    xpixels = 1000
    map.arcgisimage(service='World_Shaded_Relief',
                    xpixels = xpixels,
                    verbose= False, zorder=-3,
                    colormap="gray",
                    cmap="gray"
                    )
    if overview is True:
        map.drawmapscale(east_min+0.35,
                         north_min+0.31,
                         east_min+0.65, north_min+0.5, kmscale)
    else:
        map.drawmapscale(east_min+0.05,
                         north_min+0.01,
                         east_min+0.55, north_min+0.2, kmscale)
    parallels = np.arange(north_min, north_max, ticks)
    meridians = np.arange(east_min, east_max, ticks)
    if overview is True:
        map.drawparallels(np.round(parallels, 1), labels=[1,0,0,0], fontsize=12)
        map.drawmeridians(np.round(meridians, 1),labels=[1,1,0,1],
                          fontsize=12, rotation=45)
    else:
        map.drawparallels(parallels, labels=[1,0,0,0], fontsize=12)
        map.drawmeridians(meridians,labels=[1,1,0,1],
                          fontsize=12, rotation=45)

    if events is not None:
        for event in events:
            mt = event.moment_tensor
            if overview is False:
                if event.lat>35.845:
                    x, y = map(event.lon, event.lat)
            else:
                x, y = map(event.lon, event.lat)

            if overview is True:
                size=12
            else:
                size=20
            beachball.plot_beachball_mpl(
                        mt,
                        ax,
                        beachball_type='full',
                        size=size,
                        position=(x, y),
                        color_t=plot.mpl_color('scarletred2'),
                        linewidth=1.0,
                        zorder=1)
    if stations is not None:
        lats = [s.lat for s in stations]
        lons = [s.lon for s in stations]
        labels = ['.'.join(s.nsl()) for s in stations]
        x_station, y_station = map(lons, lats)
        map.scatter(x_station, y_station, marker="^", s=36, c="g", zorder=8)
        for k, label in enumerate(labels):
            plt.text(x_station[k], y_station[k], str(label), fontsize=12)

    if rect_lats is not None:
        import matplotlib.patches as patches
        if add_grid is True:
            for lat in rect_lats:
                for lon in rect_lons:
                    x, y = map(lon, lat)
                    rect = patches.Rectangle((x, y), 1800, 2200, linewidth=1,
                                             edgecolor='r', facecolor='none')

                    # Add the patch to the Axes
                    ax.add_patch(rect)
        max_lat_rect = np.max(rect_lats)
        min_lat_rect = np.min(rect_lats)
        max_lon_rect = np.max(rect_lons)
        min_lon_rect = np.min(rect_lons)
        xmin, ymin = map(min_lon_rect, max_lat_rect)
        xmax, ymax = map(max_lon_rect, min_lat_rect)
        width = xmax-xmin
        length = ymin-ymax
        if overview is True:
            rect = patches.Rectangle((xmin, ymax), width, length, linewidth=5,
                                     edgecolor='k', facecolor='none', zorder=7,
                                     )


            # Add the patch to the Axes
            ax.add_patch(rect)

    if preds is None and best_mts is not None:
        k = 1
        for best_mt, ev in zip(best_mts, pred_events):
            mt = ev.moment_tensor
            x, y = map(ev.lon, ev.lat)
            plt.text(x, y+0.02, str(k), fontsize=42, zorder=9, color="k")
            k = k+1
            beachball.plot_beachball_mpl(
                        mt,
                        ax,
                        beachball_type='full',
                        size=22.,
                        position=(x, y),
                        color_t=plot.mpl_color('blue'),
                        linewidth=1.0,
                        zorder=3)
    if preds is not None:
        for pred_mts, best_mt, ev in zip(preds, best_mts, pred_events):
            x, y = map(ev.lon, ev.lat)
            plot_kwargs = {
                'beachball_type': 'full',
                'size':500,
                'position': (x, y),
                'color_t': 'black',
                'edgecolor': 'black',
                'zorder':3,
                }

            beachball.plot_fuzzy_beachball_mpl_pixmap(pred_mts, ax, best_mt,
                                                      **plot_kwargs)


    plt.show()


def plot_map(stations, center, events=None, savename=None):
    from pyrocko.plot.automap import Map
    from pyrocko.example import get_example_data
    from pyrocko import model, gmtpy
    from pyrocko import moment_tensor as pmt

    gmtpy.check_have_gmt()

    # Generate the basic map
    m = Map(
        lat=center[0],
        lon=center[1],
        radius=150000.,
        width=30., height=30.,
        show_grid=False,
        show_topo=True,
        color_dry=(238, 236, 230),
        topo_cpt_wet='light_sea_uniform',
        topo_cpt_dry='light_land_uniform',
        illuminate=True,
        illuminate_factor_ocean=0.15,
        show_rivers=False,
        show_plates=False)

    # Draw some larger cities covered by the map area
    m.draw_cities()

    # Generate with latitute, longitude and labels of the stations
    lats = [s.lat for s in stations]
    lons = [s.lon for s in stations]
    labels = ['.'.join(s.nsl()) for s in stations]

    # Stations as black triangles.
    m.gmt.psxy(in_columns=(lons, lats), S='t20p', G='black', *m.jxyr)

    # Station labels
    for i in range(len(stations)):
        m.add_label(lats[i], lons[i], labels[i])

    beachball_symbol = 'd'
    factor_symbl_size = 5.0
    if events is not None:
        for ev in events:
            mag = ev.magnitude
            if ev.moment_tensor is None:
                ev_symb = 'c'+str(mag*factor_symbl_size)+'p'
                m.gmt.psxy(
                    in_rows=[[ev.lon, ev.lat]],
                    S=ev_symb,
                    G=gmtpy.color('scarletred2'),
                    W='1p,black',
                    *m.jxyr)
            else:
                devi = ev.moment_tensor.deviatoric()
                beachball_size = mag*factor_symbl_size
                mt = devi.m_up_south_east()
                mt = mt / ev.moment_tensor.scalar_moment() \
                    * pmt.magnitude_to_moment(5.0)
                m6 = pmt.to6(mt)
                data = (ev.lon, ev.lat, 10) + tuple(m6) + (1, 0, 0)

                if m.gmt.is_gmt5():
                    kwargs = dict(
                        M=True,
                        S='%s%g' % (beachball_symbol[0],
                                    (beachball_size) / gmtpy.cm))
                else:
                    kwargs = dict(
                        S='%s%g' % (beachball_symbol[0],
                                    (beachball_size)*2 / gmtpy.cm))

                m.gmt.psmeca(
                    in_rows=[data],
                    G=gmtpy.color('chocolate1'),
                    E='white',
                    W='1p,%s' % gmtpy.color('chocolate3'),
                    *m.jxyr,
                    **kwargs)
    if savename is None:
        if events is None:
            m.save('pics/stations_ridgecrest.png')
        else:
            m.save('pics/mechanisms_scedc_ridgecrest.png')
    else:
        m.save(savename)


def plot_acc_loss(history):
    acc = history.history['accuracy']
    val_acc = history.history['accuracy']
    loss = history.history['loss']
    val_loss = history.history['loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b')
    plt.title('Training accuracy')
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b')
    plt.title('Training loss')
    plt.show()


def plot_pred_mt(pred_mt):
    for pred_m in pred_mt:

        fig = plt.figure()
        axes = fig.add_subplot(1, 1, 1)
        axes.set_xlim(-2., 4.)
        axes.set_ylim(-2., 2.)
        axes.set_axis_off()

        plot.beachball.plot_beachball_mpl(
                    pred_m,
                    axes,
                    beachball_type='full',
                    size=60.,
                    position=(0, 1),
                    color_t=plot.mpl_color('scarletred2'),
                    linewidth=1.0)
        plt.show()


def plot_pred_bayesian_mt(pred_mts, best_mt=None):
    fig = plt.figure(figsize=(4., 4.))
    fig.subplots_adjust(left=0., right=1., bottom=0., top=1.)
    axes = fig.add_subplot(1, 1, 1)

    plot_kwargs = {
        'beachball_type': 'full',
        'size': 3,
        'position': (5, 5),
        'color_t': 'black',
        'edgecolor': 'black'
        }

    beachball.plot_fuzzy_beachball_mpl_pixmap(pred_mts, axes, best_mt,
                                              **plot_kwargs)
    axes.set_xlim(0., 10.)
    axes.set_ylim(0., 10.)
    axes.set_axis_off()

    plt.show()


def plot_pred_bayesian_mt_hist(pred_mts, best_mt=None, hist=True):
    fig = plt.figure(figsize=(4., 4.))
    fig.subplots_adjust(left=0., right=1., bottom=0., top=1.)
    gs = GridSpec(3, 5)
    axes = fig.add_subplot(gs[0:2, :])

    plot_kwargs = {
        'beachball_type': 'full',
        'size': 3,
        'position': (5, 5),
        'color_t': 'black',
        'edgecolor': 'black'
        }

    beachball.plot_fuzzy_beachball_mpl_pixmap(pred_mts, axes, best_mt,
                                              **plot_kwargs)
    axes.set_xlim(0., 10.)
    axes.set_ylim(0., 10.)
    axes.set_axis_off()
    plt.axis('off')
    if hist is True:
        omegas = []
        kagans = []
        for mt in pred_mts:
            omega = omega_angle(mt.m6(), best_mt.m6())
            kagan = mtm.kagan_angle(mt, best_mt)
            omegas.append(omega)
            kagans.append(kagan)

        ax = fig.add_subplot(gs[2, 1])
        sns.distplot(omegas, kde=False)
        plt.xlabel('Omega angle (°)', fontsize=12)
        plt.ylabel('#', fontsize=12)
        xmin, xmax = ax.get_xlim()
        ax.set_xticks(np.round(np.linspace(xmin, xmax, 2), 2))
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        ax = fig.add_subplot(gs[2, 3])
        sns.distplot(kagans, color="orange", kde=False)
        plt.xlabel('Kagan angle (°)', fontsize=12)
        plt.ylabel('#', fontsize=12)
        xmin, xmax = ax.get_xlim()
        ax.set_xticks(np.round(np.linspace(xmin, xmax, 2), 2))
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)


        plt.show()

    return fig


def plot_val_vs_pred_mt(pred_mt, val_mt):
    for pred_m, real_m in zip(pred_mt, val_mt):
        omega = omega_angle(real_m.m6(), pred_m.m6())
        kagan = mtm.kagan_angle(real_m, pred_m)
        fig = plt.figure()
        axes = fig.add_subplot(1, 1, 1)
        axes.set_xlim(-2., 4.)
        axes.set_ylim(-2., 2.)
        axes.set_axis_off()

        plot.beachball.plot_beachball_mpl(
                    real_m,
                    axes,
                    beachball_type='deviatoric',
                    size=60.,
                    position=(0, 1),
                    color_t=plot.mpl_color('scarletred2'),
                    linewidth=1.0)
        plot.beachball.plot_beachball_mpl(
                    pred_m,
                    axes,
                    beachball_type='deviatoric',
                    size=60.,
                    position=(1.5, 1),
                    color_t=plot.mpl_color('scarletred2'),
                    linewidth=1.0)
        plt.show()
        print("Omega Angle:", omega, "Kagan Angle:", kagan)
