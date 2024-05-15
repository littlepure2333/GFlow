import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.signal import find_peaks
from shapely.geometry import *

from concavity.utils import make_ccw, ConcaveHull


def concave_hull(coords, knn, increase_knn=1):
    """

    :param coords: np.ndarray, shapely coordinate sequence
        the coordinates of the points on which to calculate the concave hull
    :param knn: int
        the number of nearest neighbors determines how elaborate the concave boundary will be
    :param increase_knn: int
        the number by which to increase k if the previous iteration failed

    :return: shapely Polygon
    """
    return make_ccw(ConcaveHull(coords, knn, increase_knn).concave_hull)


def process_vertices(coords, angle_threshold, filter_type, convolve, get_convex, output_type):
    vecs = np.array([coords[x + 1] - coords[x] for x in range(coords.shape[0] - 1)])
    unit_vecs = np.array([i / np.linalg.norm(i) for i in vecs])

    signs = np.array( \
        [np.sign(np.cross(unit_vecs[-1], unit_vecs[0]))] + \
        [np.sign(np.cross(unit_vecs[x], unit_vecs[x + 1])) \
         for x in range(0, unit_vecs.shape[0] - 1)])

    degs = np.array([np.degrees(np.arccos(np.clip(np.dot(unit_vecs[-1], unit_vecs[0]), -1.0, 1.0)))] + \
                    [np.degrees(np.arccos(np.clip(np.dot(unit_vecs[x], unit_vecs[x + 1]), -1.0, 1.0))) \
                     for x in range(0, unit_vecs.shape[0] - 1)]
                    )

    degs = signs * degs

    if convolve:
        cov_array = np.array([0.125, 0.225, 0.3, 0.225, 0.125])
        degs = np.convolve(degs, cov_array, mode='same')

    if get_convex == False:
        degs = -degs

    if filter_type == 'peak':

        peaks, _ = find_peaks(np.append(degs , degs), height=angle_threshold)
        peaks = np.where(peaks>=degs.shape[0], peaks - degs.shape[0], peaks)
        peaks = np.unique(peaks)
    else:
        peaks = np.where(degs > angle_threshold)[0]

    return peaks, degs


def find_concave_vertices(geom, angle_threshold=0, filter_type='peak', convolve = False, get_convex=False,
                                  output_type='geopandas'):
    """
    Finds concave points along a polygon's exterior and interiors

    :param geom: shapely Polygon or MultiPolygon
    :param angle_threshold: number
        angle between two vertices, below which a point will not be considered
    :param filter_type: str
        {'all', 'peak'}
        whether to filter all vertices above the ``angle_threshold`` or locate the peak vertices, default is 'peak'
    :param convolve: boolean
        whether to smooth the angles (for finer peak detection), default is False
    :param get_convex: boolean
        whether to get convex points instead of concave, default is False
    :param output_type: str
        {"geopandas" , "list"}
         geopandas dataframe (default) or a list of points and angles

    :return: geopandas dataframe (default) or a tuple of points and angles
    """

    if output_type == 'geopandas':

        concave_points = gpd.GeoDataFrame()
    else:
        concave_points = [[], []]

    if isinstance(geom, MultiPolygon):
        if output_type == 'geopandas':

            frames = []
            for g in geom:
                frames.append(find_concave_vertices(g, angle_threshold, filter_type,convolve, get_convex, output_type))

            concave_points = pd.concat(frames)
            return concave_points

        else:
            points = []
            angles = []
            for g in geom:
                pts, angs = find_concave_vertices(g, angle_threshold, filter_type, convolve, get_convex, output_type)
                points += pts
                angles += angs
            return points, angles


    else:

        geom = make_ccw(geom)

        for coord_seq in [np.array(i.coords) for i in geom.interiors] + [np.array(geom.exterior.coords)]:
            peaks, degs = process_vertices(coord_seq, angle_threshold, filter_type, convolve, get_convex, output_type)

            if output_type == 'geopandas':

                concave_points = pd.concat([concave_points,
                                            gpd.GeoDataFrame({'geometry': [Point(i) for i in coord_seq[:-1][peaks]],
                                                              'angle': degs[peaks]})])
            else:
                concave_points[0] += [Point(i) for i in coord_seq[:-1][peaks]]
                concave_points[1] += degs[peaks].tolist()

        return concave_points


def find_convex_vertices(geom, angle_threshold, filter_type,convolve = False, get_concave=False, output_type='geopandas'):
    """
    Finds convex points along a polygon's exterior and interiors

    :param geom: shapely Polygon or MultiPolygon
    :param angle_threshold: number
        angle between two vertices, below which a point will not be considered
    :param filter_type: str
        {'all', 'peak'}
        whether to filter all vertices above the ``angle_threshold`` or locate the peak vertices, default is 'peak'
    :param convolve: boolean
        whether to smooth the angles (for finer peak detection), default is False
    :param get_convex: boolean
        whether to get convex points instead of concave, default is False
    :param output_type: str
        {"geopandas" , "list"}
         geopandas dataframe (default) or a list of points and angles

    :return: geopandas dataframe (default) or a tuple of points and angles
    """
    if get_concave == False:
        get_convex = True
    else:
        get_convex = False

    return find_concave_vertices(geom, angle_threshold, filter_type,convolve,  get_convex, output_type)


def plot_concave_hull(coords, concave_poly, figsize = (10,10)):
    """

    :param coords: numpy.ndarray, shapely coordinate sequence
        the points the concave hull was calculated upon
    :param concave_poly: shapely Polygon
        the output concave hull polygon
    :param figsize: tuple
        figure size
    :return: None
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize)
    legend_elements = []

    gpd.GeoSeries([MultiPoint(coords).convex_hull]).plot(ax=ax,
                                                         color='w',
                                                         edgecolor='#5ac5bc',
                                                         linewidth=5)
    legend_elements += [Patch(alpha=0.5,
                              facecolor='w',
                              edgecolor='#5ac5bc',
                              linewidth=2,
                              label='convex polygon')]

    gpd.GeoSeries([concave_poly]).plot(ax=ax,
                                       alpha=0.5,
                                       color='#ee65a3',
                                       edgecolor='r',
                                       linewidth=5)

    legend_elements += [Patch(alpha=0.5,
                              facecolor='#ee65a3',
                              edgecolor='r',
                              linewidth=2,
                              label='concave polygon')]

    gpd.GeoSeries([Point(c) for c in coords]).plot(ax=ax,
                                                   color='purple'
                                                   )

    legend_elements += [Line2D([0], [0], markersize=8, color='w',
                               markerfacecolor='purple',
                               marker='o',
                               label='points')]

    ax.legend(handles=legend_elements, )
    plt.show()


def plot_vertices(geom, concave_vertices_df=None, convex_vertices_df = None, figsize = (10,10)):
    """

        :param coords:  shapely Polygon
            the polygon the concave/convex vertices were calculated on
        :param concave_vertices_df: geopandas GeoDataFrame
            the output concave vertices, default is None
        :param concave_vertices_df: geopandas GeoDataFrame
            the output convex vertices, default is None
        :param figsize: tuple
            figure size
        :return: None
        """
    from matplotlib.lines import Line2D
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize)
    legend_elements = []
    gpd.GeoSeries([geom]).plot(ax=ax, color='#ee65a3', alpha=0.5, edgecolor='r')

    if concave_vertices_df is not None:
        concave_vertices_df.plot(ax=ax, color='#5ac5bc')
        legend_elements += [Line2D([0], [0], markersize=8, color='w',
                                   markerfacecolor='#5ac5bc',
                                   marker='o',
                                   label='concave vertices')]

    if convex_vertices_df is not None:
        convex_vertices_df.plot(ax=ax, color='purple')
        legend_elements += [Line2D([0], [0], markersize=8, color='w',
                                   markerfacecolor='purple',
                                   marker='o',
                                   label='convex vertices')]

    ax.legend(handles=legend_elements, )
    plt.show()