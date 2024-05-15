import numpy as np
from scipy.ndimage import gaussian_filter1d
from shapely.geometry import *
from shapely.ops import polygonize
import warnings
import rtree
from PIL import Image, ImageDraw


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



def make_valid_polygon(geom, precision=2):
    if geom.is_valid == True or isinstance(geom, (Polygon, MultiPolygon)) == False:
        return geom

    if isinstance(geom, Polygon):
        area = geom.area
        buff_geom = geom.buffer(0)
        buff_area = buff_geom.area
        if round(area, precision) == round(buff_area, precision):
            return buff_geom
        else:

            exterior = geom.exterior
            interiors = geom.interiors
            exterior = exterior.intersection(exterior)
            interiors = [MultiPolygon(polygonize(i.intersection(i)))[0].exterior for i in interiors]
            result = MultiPolygon([Polygon(i.exterior, interiors) for i in polygonize(exterior)])
            return result
    elif isinstance(geom, MultiPolygon):
        result = [make_valid_polygon(poly) for poly in geom]
        result = MultiPolygon(flatten([[i for i in j] if isinstance(j, MultiPolygon) else j for j in result]))
        if len(result) == 1:
            return result[0]
        else:
            return result


def make_ccw(geom):
    if isinstance(geom, (Polygon, MultiPolygon)) == False:
        warnings.warn("Geometry is not a Polygon or MultiPolygon - returning the original geometry")
        return geom


    if isinstance(geom, Polygon):
        if geom.is_empty == True:
            return geom
        if geom.exterior.is_ccw:
            return geom
        else:
            exterior = LinearRing(geom.exterior.coords[::-1])
            interiors = geom.interiors
            ## make the interior cw and not ccw
            interiors = [LinearRing(i.coords[::-1]) if i.is_ccw == True else i for i in interiors]
            result = Polygon(exterior, interiors)
            return result
    elif isinstance(geom, MultiPolygon):
        result = [make_ccw(poly) for poly in geom]
        result = MultiPolygon(flatten([[i for i in j] if isinstance(j, MultiPolygon) else j for j in result]))
        if len(result) == 1:
            return result[0]
        else:
            return result

def flatten(container):
    for i in container:
        if isinstance(i, (list, tuple)):
            for j in flatten(i):
                yield j
        else:
            yield i



class ConcaveHull(object):
    def __init__(self,
                 coords,
                 knn,
                 increase_knn=1):


        self.coords_orig = coords
        self.knn_orig = knn
        self.increase_knn_orig = increase_knn
        self._concave_hull(self.coords_orig, self.knn_orig, self.increase_knn_orig)


    def get_concave_hull(self):
        return self.concave_hull

    def _set_params(self, current_idx, knn, new_dir, current_point ):

        # get nearest neighbors
        self.idx = np.array(list(self.rt_idx.nearest(self.c_bounds[current_idx], knn)))
        self.idx = self.idx[self.idx != current_idx]
        self.points = self.coords[self.idx]

        # calculate the direction between the current point and its neighbors
        self.dir_vecs = np.array([x - self.coords[current_idx] for x in self.points])
        self.dir_vecs = np.array([i / np.linalg.norm(i) for i in self.dir_vecs])

        # calculate the angles between the current direction and the neighbors
        self.signs = np.array([np.sign(np.cross(new_dir, i)) for i in self.dir_vecs])
        self.angles = np.array([np.degrees(np.arccos(np.clip(np.dot(new_dir, i), -1.0, 1.0))) for i in self.dir_vecs])
        self.angles[self.signs == 1] = 360 - self.angles[self.signs == 1]
        self.angles = np.array([90 - x if x < 90 else 360 + 90 - x for x in self.angles])

        # calculate the distances between the current point and all other points
        self.dists = np.array([np.linalg.norm(current_point - i) for i in self.points])

        # a queue of neighbors sorted on min angle, max dist
        self.s_points = np.array([self.angles, self.dists, self.idx, np.arange(0, self.idx.shape[0])]).T
        self.s_points = self.s_points[np.lexsort((-self.s_points[:, 1], self.s_points[:, 0]))]

        # take the top neighbor
        self.new_idx = int(self.s_points[0][2])

    def _concave_hull(self, coords, knn, increase_knn):

        self.coords = np.array(coords)
        self.knn = knn
        self.increase_knn = increase_knn



        # triangle case
        if np.unique(self.coords, axis = 0).shape[0] < 3:
            self.concave_hull =  Polygon()
        elif np.unique(self.coords, axis = 0).shape[0] == 3:
            self.concave_hull = MultiPoint(self.coords).convex_hull
          # no concave shape could be drawn
        elif self.knn == np.unique(self.coords, axis = 1).shape[0] - 1:
            self.concave_hull = MultiPoint(self.coords).convex_hull

        else:


            if self.knn_orig >= self.coords.shape[0]-1:
                self.knn =  self.coords.shape[0] - 2

            # bounds array for rtree index
            self.c_bounds = np.hstack([self.coords, self.coords])

            # rtree index
            self.rt_idx = rtree.index.Index()
            [self.rt_idx.insert(i[0], (i[1])) for i in enumerate(self.c_bounds)]

            # set initial params
            self.start_knn = self.knn
            self.line = []
            self.min_y = self.coords[:, 1].min()
            self.min_y_idx = np.where(self.coords[:, 1] == self.min_y)
            self.first_idx = self.min_y_idx[0][0]
            self.line.append(self.coords[self.first_idx])
            self.first_point = self.coords[self.first_idx]

            # set params
            self._set_params( self.first_idx, self.knn, np.array([.0, 1.]), self.first_point)
            self.current_idx = self.new_idx
            self.current_point = self.coords[self.current_idx]
            self.new_dir = self.dir_vecs[int(self.s_points[0][3])]
            self.line.append(self.current_point)

            # delete the first point from the rtree index
            self.rt_idx.delete(self.first_idx, self.c_bounds[self.first_idx])

            # distance when the first point will be re-added to the index
            self.check_dist = True
            self.max_dist = max(self.dists)

            while self.current_idx != self.first_idx:
                self.test_int = True
                count = 0
                self.knn = self.start_knn

                # set the queue
                self._set_params(self.current_idx, self.knn, self.new_dir, self.current_point)
                while self.test_int == True:

                    # this stops the iteration
                    if self.new_idx == self.first_idx:
                        self.test_int = False
                    else:

                        # set the new direction
                        self.new_dir = self.dir_vecs[int(self.s_points[count][3])]
                        count += 1


                        # check if the new line will create self intersection with the existing line
                        l = LineString(self.line)
                        relate = l.relate(LineString([self.current_point,
                                                      self.coords[self.new_idx]]))

                        # if it doesn't, stop the inner loop and add it to the line
                        if relate == 'FF1F00102':
                            self.test_int = False

                        else:
                            self.test_int = True
                            self.rt_idx.delete(self.new_idx, self.c_bounds[self.new_idx])

                            # if the queue is empty, re-polulate it
                            if count == self.s_points.shape[0]:
                                self._set_params(self.current_idx, self.knn, self.new_dir, self.current_point)
                                count = 0
                            # continue the loop and delete the candidate point from the rtree index


                # delete the current point from the rtree index
                self.rt_idx.delete(self.current_idx, self.c_bounds[self.current_idx])

                # set the candidate point as the current point
                self.current_idx = self.new_idx
                self.current_point = self.coords[self.current_idx]

                # add the first point to the r-tree index after the safety distance has been reached
                if self.check_dist:
                    if np.linalg.norm(self.first_point - self.current_point) >= self.max_dist:
                        self.rt_idx.add(self.first_idx, self.c_bounds[self.first_idx])
                        self.check_dist = False

                # append the new point to the line
                self.line.append(self.current_point)

            # the bottom two cases are fails due to some points being too far and the number of neighbors being too small
            # re-calculte, incrementing knn
            if len(self.line) < 4:

                return self._concave_hull(self.coords_orig, self.knn + self.increase_knn, self.increase_knn)
            else:
                poly = Polygon(self.line)
                if poly.buffer(0.1).contains(MultiPoint(self.coords)) and poly.is_valid:

                    self.concave_hull = poly
                else:
                    return self._concave_hull(self.coords_orig, self.knn + self.increase_knn, self.increase_knn)