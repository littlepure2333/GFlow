import warnings
import torch
from concave_hull import concave_hull
from shapely import LineString, MultiLineString, MultiPolygon
from shapely.geometry import Polygon
from PIL import Image, ImageDraw
import numpy as np
from scipy.ndimage import gaussian_filter1d

def polygon_to_mask(polygon, width, height):
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon(polygon.exterior.coords, outline=1, fill=1)
    del draw

    return np.array(mask)

def gaussian_smooth(coords, sigma=2, num_points_factor=2):

    coords = np.array(coords)
    x, y = coords.T
    xp = np.linspace(0, 1, coords.shape[0])
    interp = np.linspace(0, 1,coords.shape[0] * num_points_factor)
    x = np.interp(interp, xp, x)
    y = np.interp(interp, xp, y)
    x = gaussian_filter1d(x, sigma, mode='wrap')
    y = gaussian_filter1d(y, sigma, mode='wrap')
    return x, y

def gaussian_smooth_geom(geom, sigma=2, num_points_factor=2):
    """
    :param geom: a shapely LineString, Polygon, MultiLineString ot MultiPolygon
    :param sigma: standard deviation for Gaussian kernel
    :param num_points_factor: the number of points determine the density of vertices  - resolution
    :return: a smoothed shapely geometry
    """

    if isinstance(geom, (Polygon, LineString)):
        x, y = gaussian_smooth(geom.exterior.coords)

        if type(geom) == Polygon:
            x = np.append(x, x[0])
            y = np.append(y, y[0])
            if len(list(geom.interiors)) > 0:
                l = []
                l.append(list(zip(x, y)))

                for interior in list(geom.interiors):
                    x, y = gaussian_smooth(interior)
                    l.append(list(zip(x, y)))
                return Polygon([item for sublist in l for item in sublist])
            else:

                return Polygon(list(zip(x, y)))
        else:
            return LineString(list(zip(x, y)))
    elif isinstance(geom, (MultiPolygon, MultiLineString)):
        list_ = []
        for g in geom:
            list_.append(gaussian_smooth_geom(g, sigma, num_points_factor))

        if type(geom) == MultiPolygon:

            return MultiPolygon(list_)
        else:
            return MultiLineString(list_)
    else:
        warnings.warn('geometry must be LineString, Polygon, MultiLineString or MultiPolygon, returning original geometry')
        return geom



class FastConcaveHull2D:
    def __init__(self, points, sigma=2, num_points_factor=5):
        self.points = points

        # judge if the points are in numpy format, if not, convert them to numpy
        if isinstance(points, torch.Tensor):
           self.points = points.detach().cpu().numpy()
        
        # calculate the convex hull
        points = concave_hull(self.points)
        self.hull = Polygon(points)
        if sigma > 0:
            self.hull = gaussian_smooth_geom(self.hull, sigma=sigma, num_points_factor=num_points_factor)
    
    def area(self):
        return self.hull.area

    def mask(self, width, height):
        mask = polygon_to_mask(self.hull, width, height)

        return mask