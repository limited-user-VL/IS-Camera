"""
This project is used to evaluate the output of Vitrealab light chips.
The goal is to extract the tilt angles (tilt_x, tilt_y) and beam
divergence (beam_div_x, beam_div_y) for each beam in the beam array.

Author: Rui

class Tomography: contains all the acquired images;

class CrossSection: contains image and rotated image and z-coordinate
    of respective cross-section;
class Beam: contains (id_x, id_y) and (tilt_x, tilt_y) and (beam_div_x, beam_div_y);

"""
#standard library imports
import pickle
import numpy as np
import os
import skimage
from skimage import transform as t
from skimage import exposure as e
from skimage import filters as f
from skimage import feature
from skimage.measure import regionprops #for center of mass
from tqdm import tqdm
import time
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.patches import Rectangle

#3rd party imports

#local library imports


def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time} seconds to complete.")
        return result
    return wrapper

def find_centroid(grayscale_image):
    # Ensure the image is a numpy array
    image_array = np.array(grayscale_image)

    # Create a grid of coordinates
    y_indices, x_indices = np.indices(image_array.shape)

    # Calculate the sum of intensities
    total_intensity = np.sum(image_array)

    # Calculate the weighted sum of coordinates
    centroid_x = np.sum(x_indices * image_array) / total_intensity
    centroid_y = np.sum(y_indices * image_array) / total_intensity

    return centroid_x, centroid_y

# Define the Gaussian function
def gaussian(x, mean, standard_deviation):
    return 1 * np.exp( - (x - mean)**2 / (2 * standard_deviation ** 2))

def get_beam_specs(roi_img, roi_width, debug = False):
    """
    Given a 2D array with a region of interest (ROI) around a beam,
    fit a gaussian beam to the marginal X and marginal Y distributions
    and retrieve the middle point for X and Y and the sigma of the
    gaussian fit..

    :param roi_img: (2D array) - region of interest around beam
    :param roi_width (int) - width of region of interest in pixel units;
    :return: row_mu, col_mu, sigma
    """

    #Project intensity map of ROI into x and y --> I(x) = roi_sum_col, I(y) = roi_sum_row
    roi_sum_row_arr = np.sum(roi_img, axis=0)
    roi_sum_col_arr = np.sum(roi_img, axis=1)

    #Normalise I(x) and I(y)
    roi_sum_row_arr = roi_sum_row_arr - np.min(roi_sum_row_arr)
    roi_sum_row_arr = roi_sum_row_arr / np.max(roi_sum_row_arr)
    roi_sum_col_arr = roi_sum_col_arr - np.min(roi_sum_col_arr)
    roi_sum_col_arr = roi_sum_col_arr / np.max(roi_sum_col_arr)

    row_idx_arr = np.arange(0, len(roi_sum_row_arr))
    col_idx_arr = np.arange(0, len(roi_sum_col_arr))

    try:
        mean_0 = roi_width/2
        std_0 = roi_width/5
        p0 = [mean_0, std_0]

        #fit sum of rows
        popt_row, _ = curve_fit(gaussian, row_idx_arr, roi_sum_row_arr, p0 = p0)
        row_mu = int(popt_row[0])
        row_sigma = int(popt_row[1])

        # fit sum of cols
        popt_col, _ = curve_fit(gaussian, col_idx_arr, roi_sum_col_arr, p0 = p0)
        col_mu = int(popt_col[0])
        col_sigma =  int(popt_col[1])

        #sigma as average of sigma_x and sigma_y
        sigma = 0.5*(row_sigma + col_sigma)

    except (RuntimeError, ValueError):  # if the fit fails
        row_mean = int(roi_width / 2)
        col_mean = int(roi_width / 2)
        popt_row = [mean_0, std_0]
        popt_col = [mean_0, std_0]
        sigma = int(roi_width / 5)

    if debug:
        plt.figure()
        plt.plot(row_idx_arr, roi_sum_row_arr, ".", color = "k")
        y_fit = gaussian(row_idx_arr, *popt_row)
        plt.plot(row_idx_arr, y_fit, label = "row")
        plt.xlabel("[px]")

        plt.plot(col_idx_arr, roi_sum_col_arr,".", color = "k")
        y_fit = gaussian(row_idx_arr, *popt_col)
        plt.plot(row_idx_arr, y_fit, label = "col")
        plt.legend()


    return row_mu, col_mu, row_sigma, col_sigma

def get_roi(image, center_x, center_y, roi_width_x, roi_width_y):
    """
    Given a 2D array, the function returns a region of interest within the 2D array, specified
    by the center coordinates (x,y) and the width of the region of interest in x and y;
    :param image: 2d numpy array
    :param center_x: int
    :param center_y: int
    :param roi_width_x: int
    :param roi_width_y: int
    :return: roi: 2d numpy array
    """
    roi = image[int(center_x - roi_width_x / 2.): int(center_x + roi_width_x / 2.),
          int(center_y - roi_width_y / 2.): int(center_y + roi_width_y / 2.)]
    return roi


class Tomography:
    def __init__(self, filename, shape, roi_width = 100):
        """
        Instantiate Tomography measurement.

        Args:
        1. Filename (str): File location and name
        2. Shape (tuple): (n_rows, n_cols), e.g (3,2)

        """
        self.filename = filename
        self.directory = os.getcwd()
        self.shape = shape
        self.roi_width = roi_width

        # updated with method load_data
        self.cross_sect_image_l = None  # list with cross section images
        self.cross_sect_z_l = None  # list with cross section z coord
        self.cross_sect_l = list()  # list with cross section objects

        self.n_sections = None  # number of cross sections
        self.z_diff = None  # z difference between cross_sections;

        # geometric parameters --> init_coords, complete_coords
        self.pixel_size = 3.45 * 10 ** -6  # 3.45um, for IS DMM 37UX256-ML-37U

        #initialise beams
        self.beam_l = [[Beam(id_x, id_y) for id_y in range(self.shape[1])] for id_x in range(self.shape[0])] #list with all beam objects;

        #updated with manual investigation of beam trajectory (after tomo.complete_coords)
        self.max_z_fit = None
        self.max_z_idx_fit = None


    def __str__(self):
        return f"""Tomography measurement:\n
        - Filename = {self.filename}\n
        - Number of beam rows = {self.shape[0]:.0f}\n
        - Number of beam cols = {self.shape[1]:.0f}\n
        - Z-spacing: {self.z_diff:.3f}mm\n
        - Number of cross sections: {self.n_sections:.0f}
        """

    def __repr__(self):
        return f"Tomography object, shape = {self.shape}, filename = {self.filename}"

    def load_data(self):
        """
        Load:
        1. List with images of the cross sections
        2. List with values of z coordinate of each cross section
        """

        with open(self.filename, "rb") as file:
            data_dict = pickle.load(file)

        self.cross_sect_image_l = data_dict["img_store"]
        self.cross_sect_z_l = data_dict["coord_store"]
        self.n_sections = len(self.cross_sect_image_l)
        self.z_diff = np.diff(np.array(self.cross_sect_z_l)).mean()
        self.cross_sect_l = [[] for _ in range(self.n_sections)]

        for i in range(self.n_sections):  # create cross section objects
            z_i = self.cross_sect_z_l[i]
            shape_i = self.shape
            image_i = self.cross_sect_image_l[i]
            cross_i = Cross_Section(z_i, shape_i, image_i)
            self.cross_sect_l[i] = cross_i  # append cross_section object

        print("Loaded")
        print(self)

    def find_rot_spacing(self, angle_min = 0, angle_max = 90, angle_step=0.5):
        """
        Uses image of the cross-section with lowest z value to extract:
        1. Rotation angle, using the first cross-section (lowest z-value)
        2. Grid spacing;
        3. The rotation angle, rotated images and spacing are written to the cross-section objects

        Parameters:
            angle_step - incremental difference in angle of rotation;
        """

        # 1. Extract rotation angle, using the first cross-section (lowest z-value)
        print("Extracting rotation angle for the lowest z cross section.")
        cross_sect_i = self.cross_sect_l[0]
        opt_angle = cross_sect_i.find_rot(angle_min=angle_min, angle_max=angle_max,
                                          angle_step=angle_step, plot=False)
        print(f"Optimal rotation angle = {opt_angle:.2f}deg")

        # 2. Extract grid spacing;
        print("Extracting the grid spacing")
        cross_sect_i = self.cross_sect_l[0]
        peak_arr = cross_sect_i.find_peaks(nrows=3, ncols=10, min_distance=100)

        # label beams - line_id, col_id
        kmeans_rows = KMeans(n_clusters=3)
        kmeans_rows.fit(peak_arr[:, 0].reshape(-1, 1))  # kmeans, 1 cluster per row
        coords_rows = kmeans_rows.cluster_centers_
        mean_delta_x = np.mean(np.diff(np.sort(coords_rows, axis=0), axis=0))  # spacing between rows
        spacing = mean_delta_x
        print(f"Average spacing [px] between beams = {spacing:.2f}")

        # 3. The rotation angle, rotated images and spacing are written to the cross-section objects
        print("Updating the rotation angle and rotated image for each cross section.")
        self.spacing_px = spacing
        self.spacing_mm = spacing * self.pixel_size * 10**3
        for cross_sect_i in tqdm(self.cross_sect_l):
            cross_sect_i.rot_angle = opt_angle
            cross_sect_i.image_rot = t.rotate(cross_sect_i.image, cross_sect_i.rot_angle)
            cross_sect_i.spacing_px = self.spacing_px
            cross_sect_i.spacing_mm = self.spacing_mm

    def init_coords(self):
        """
        Finds the coordinates of the beams on the lowest cross-section.
        The coordinates are then pushed into:
         1. CrossSection.beam_coord_l
         2. Beam_i.beam_coord_l

        :return:
        """
        # call tomo.coord_init()
        exp_num_peaks = self.shape[0] * self.shape[1]
        cross_0 = self.cross_sect_l[0]
        spacing = int(cross_0.spacing_px)
        image_i = self.cross_sect_l[0].image_rot

        #find peaks
        peak_arr = feature.peak_local_max(image_i, num_peaks=exp_num_peaks, min_distance=int(spacing * 0.8))
        peak_sorted_arr = [[[None for k in range(3)] for j in range(self.shape[1])] for i in range(self.shape[0])]

        #sort peaks to match the index of the beams
        min_x = np.min(peak_arr[:, 0])
        min_y = np.min(peak_arr[:, 1])
        spacing = self.spacing_px
        for id_x in range(self.shape[0]):
            for id_y in range(self.shape[1]):
                for k in range(3):
                    coord_x = min_x + id_x * spacing #in pixel units
                    coord_y = min_y + id_y * spacing #in pixel units;

                    d = np.sum(np.abs(peak_arr - np.array([coord_x, coord_y])), axis=1)
                    idx_min = np.argmin(d, axis=0)
                    coord_i = np.concatenate((peak_arr[idx_min], np.array(self.cross_sect_z_l[0]).reshape(-1)),axis=0)
                    peak_sorted_arr[id_x][id_y][k] = coord_i

        peak_sorted_arr = np.array(peak_sorted_arr)

        # pass peak coords to cross section objects
        cross_0.beam_coord_l = peak_sorted_arr

        # pass peak coords into each beam, listed in self.beam_l
        for id_x in range(self.shape[0]):
            for id_y in range(self.shape[1]):
                self.beam_l[id_x][id_y].beam_coord_l = [[] for _ in range(self.n_sections)] #init beam.beam_coord_l
                self.beam_l[id_x][id_y].beam_coord_l[0] = peak_sorted_arr[id_x, id_y, 0]
                self.beam_l[id_x][id_y].beam_width_l = [[] for _ in range(self.n_sections)] #init beam.beam_coord_l

                self.beam_l[id_x][id_y].roi_l = [[] for _ in range(self.n_sections)] #initialise beam_i.roi_l
        print("Coordinates of beam in first layer were determined.")

    def complete_beam_coords(self, id_x, id_y, debug = False):
        beam_i = self.beam_l[id_x][id_y]

        for id_z in range(self.n_sections):
            #define initial_guess for center of ROI (region of interest)
            if id_z == 0:
                coord_x, coord_y, coord_z = beam_i.beam_coord_l[id_z]
            else:
                coord_x, coord_y, coord_z = beam_i.beam_coord_l[id_z-1]

            #get beam roi, based on coordinates of beam in lower section
            image_i = self.cross_sect_l[id_z].image_rot
            roi_i = get_roi(image_i, coord_x, coord_y,
                            roi_width_x = self.roi_width,
                            roi_width_y = self.roi_width)
            roi_i = gaussian_filter(roi_i, sigma=self.roi_width / 10)  # smoothen ROI for pixel noise

            # establish new coordinates of beam in current z_id, with gaussian fit
            #row_mu corresponds to a sum over rows --> I(y)
            #col_mu corrsponds to a sum over cols --> I(x)
            row_mu, col_mu, row_sigma, col_sigma = get_beam_specs(roi_i, self.roi_width, debug=debug)
            coord_x = int(coord_x + (col_mu - self.roi_width / 2))
            coord_y = int(coord_y + (row_mu - self.roi_width / 2))
            coord_z = self.cross_sect_z_l[id_z]

            if debug:
                plt.figure()
                plt.title(f"z = {coord_z:.2f}mm")
                plt.imshow(roi_i, origin="lower")

            # update beam.roi_l and beam.beam_coord_l
            beam_i.beam_coord_l[id_z] = np.array([coord_x, coord_y, coord_z])
            beam_i.beam_width_l[id_z] = np.array([row_sigma, col_sigma])
            beam_i.roi_l[id_z] = roi_i

        if debug:
            print(f"The coordinates of beam ({id_x:.0f},{id_y:.0f}) have been determined for all cross-sections.")

    def complete_all_beams_coords(self, debug = False):
        """
        Calls complete_beam_coords iteratively to cover all beams on the chip
        updates:
         1. beam_i.beam_coord_l
         2. beam_i.beam_width_l
         3. beam_i.roi_l

        :param debug:
        :return:
        """

        for id_x in tqdm(range(self.shape[0])):
            for id_y in range(self.shape[1]):
                self.complete_beam_coords(id_x, id_y, debug = debug)

    def complete_coords_obsolete(self, debug = False):
        """
        For each beam, determine the (x,y,z) coordinates at the different cross-sections.
        v3: The coordinates are determined by the point of max brightness within each region of interest (ROI)
        v4: The coordinates are determined by a gaussian fit to the marginal intensity distributions, along x and y

        :return:
        """

        for id_x in tqdm(range(self.shape[0])):
            for id_y in range(self.shape[1]):
                roi_width = self.roi_width
                # for an arbitrary beam, extract the coords in the cross section z_1

                if debug: print(id_x, id_y, "\n")
                beam_i = self.beam_l[id_x][id_y]

                for id_z in range(self.n_sections - 1):
                    # beam info in cross section i
                    coord_x, coord_y, coord_z = self.beam_l[id_x][id_y].beam_coord_l[id_z]
                    if id_z == 0: #just for the first layer
                        image_i = self.cross_sect_l[id_z].image_rot
                        roi_i = get_roi(image_i, coord_x, coord_y, roi_width, roi_width)

                        #correct coordinates determined with init_coords method
                        row_mean, col_mean = get_mean_idx(roi_i, roi_width)
                        coord_x = int(row_mean + (coord_x - roi_width / 2))
                        coord_y = int(col_mean + (coord_y - roi_width / 2))
                        coord_z = self.cross_sect_z_l[id_z]

                        beam_i.beam_coord_l[id_z] = np.array([coord_x, coord_y, coord_z])
                        beam_i.roi_l[id_z] = roi_i

                    # beam info in cross section i+1
                    image_ip1 = self.cross_sect_l[id_z + 1].image_rot
                    roi_ip1 = get_roi(image_ip1, coord_x, coord_y, roi_width, roi_width)
                    roi_ip1 = gaussian_filter(roi_ip1, sigma=roi_width / 10)  # smoothen ROI for pixel noise

                    #establish new coordinates
                    row_mean, col_mean = get_mean_idx(roi_ip1, roi_width, debug = debug)

                    coord_x = int(coord_x + (row_mean - roi_width / 2))
                    coord_y = int(coord_y + (col_mean  - roi_width / 2))
                    coord_z = self.cross_sect_z_l[id_z + 1]

                    # update beam_i information (roi_l and beam_coord_l)
                    roi_ip1 = get_roi(image_ip1, coord_x, coord_y, roi_width, roi_width)
                    roi_ip1 = gaussian_filter(roi_ip1, sigma=roi_width / 10)  # smoothen ROI for pixel noise
                    if debug:
                        plt.figure()
                        plt.title(f"z = {coord_z:.2f}mm")
                        plt.imshow(roi_ip1, origin = "lower")

                    #update beam.roi_l and beam.beam_coord_l
                    beam_i.beam_coord_l[id_z +1 ] = np.array([coord_x, coord_y, coord_z])
                    beam_i.roi_l[id_z + 1] = roi_ip1

                if debug:
                    print(f"The coordinates of beam ({id_x:.0f},{id_y:.0f}) have been determined for all cross-sections.")

    def plot_cross_section(self, id_z):
        """
        Plot cross sections together with ROIs, for a given cross-section;

        :param id_z: identifies cross-section. From 0 to self.n_sections;
        :return: f, ax
        """

        # find rotated image of cross section
        image_i = self.cross_sect_l[id_z].image_rot

        # find beam_coordinates
        rois_in_cross_section = [[ [] for _ in range(self.shape[1])] for _ in range(self.shape[0])]

        for id_x in range(self.shape[0]):
            for id_y in range(self.shape[1]):
                beam_i = self.beam_l[id_x][id_y]
                coords = beam_i.beam_coord_l[id_z]
                rois_in_cross_section[id_x][id_y] = coords

        # find rois_width
        roi_width = self.roi_width

        f, ax = plt.subplots()
        plt.imshow(image_i, origin = "lower")
        plt.xlabel("col [px]")
        plt.ylabel("row [px]")

        for id_x in range(self.shape[0]):
            for id_y in range(self.shape[1]):
                row, col, z = rois_in_cross_section[id_x][id_y]

                # Create a rectangle patch
                rect = Rectangle((int(col-roi_width/2), int(row - roi_width/2)),
                                         roi_width, roi_width, linewidth=2, edgecolor='r', facecolor='none')

                # Add the rectangle to the Axes
                ax.add_patch(rect)
                plt.title(f"id_z = {id_z}, z = {self.cross_sect_z_l[id_z]:.2f}mm")
        plt.tight_layout()

        return f, ax

    def set_max_z(self, max_z):
        """
        Establish max z-value [mm] of cross sections to be used in fitting to extract tilt_x, tilt_y, div_x, div_y
        of the single beams.

        Updated max_z_fit and max_z_idx_fit and beam_i.max_z_fit and beam_i.max_z_idx_fit for all beams
        :param max_z:
        :return:
        """
        self.max_z_fit = max_z
        self.max_z_idx_fit = np.argmin(np.abs(np.array(self.cross_sect_z_l)-self.max_z_fit))
        print("Tomo.max_z_fit and Tomo.max_z_idx_fit have been updated.")

        #pass information down to all the beams;
        for id_x in range(self.shape[0]):
            for id_y in range(self.shape[1]):
                beam_i = self.beam_l[id_x][id_y]
                beam_i.max_z_fit = self.max_z_fit
                beam_i.max_z_idx_fit = self.max_z_idx_fit

        print("beam_i.max_z_fit and beam_i.max_z_idx_fit have been updated in all beams.")

    def find_dir_cos(self, debug):
        """
        Iterates over beams and calls method of Beam class find_dir_cos.
        The direction cosine attributes of the Beam instances are determined.

        :param debug:
        :return: None
        """
        if self.max_z_idx_fit is None:
            print("Please check beam trajectories manually and update the maximum z value to be used in fitting.")
            return None

        else:
            for id_x in range(self.shape[0]):
                for id_y in range(self.shape[1]):
                    beam_i = self.beam_l[id_x][id_y]
                    beam_i.find_dir_cos(self.pixel_size, debug = debug)


class Cross_Section:
    def __init__(self, z_coord, shape, image):
        self.z_coord = z_coord
        self.shape = shape  # [n_rows x n_cols]
        self.image = image

        self.spacing_px = None
        self.spacing_mm = None
        self.rot_angle = None
        self.image_rot = None  # The main reference frame wil be rotated wrt the original

        self.beam_coord_l = None #coordinate of beam centroids at the same z

    def __str__(self):
        return f"Cross section at z={self.z_coord:.2f}mm, with {self.shape[0]} rows and {self.shape[1]} cols."

    def __repr__(self):
        return f"CrossSection, z = {self.z_coord:.2f}, shape = {self.shape} "

    def simple_plot(self, rotated=True):
        f, ax = plt.subplots()
        #x_range_mm = self.image.shape[1] * 3.45 * 10 ** -3
        #y_range_mm = self.image.shape[0] * 3.45 * 10 ** -3

        if rotated:
            if self.image_rot is not None:
                image = self.image_rot
                plot = True
            else:
                print("Rotated image has not been generated yet. Run find_geometry from Tomography class.")
                plot = False
        else:
            image = self.image
            plot = True

        plt.imshow(image, origin="lower",
                   #extent=[0, x_range_mm, 0, y_range_mm],
                   interpolation = "none")
        plt.xlabel("Y [px]")
        plt.ylabel("X [px]")
        plt.title(f"Z = {self.z_coord:.2f}mm")
        return f, ax

    def find_rot(self, angle_min = 0, angle_max = 90, angle_step=0.5, plot=False):
        """
        Rotates cross section image, such that rows are horizontal
        and columns are vertical.
        """

        max_arr = list()
        angle_arr = np.arange(angle_min, angle_max, angle_step)

        # image is rotated by different angles
        for angle_i in tqdm(angle_arr):
            image_t = t.rotate(self.image, angle_i)  # rotate
            image_t = e.equalize_adapthist(image_t)
            # plt.imshow(image_t)

            horizontal_sum = np.sum(image_t, axis=1)  # sums along rows
            horizontal_max = np.max(horizontal_sum)
            max_arr.append(horizontal_max)

        angle_opt = angle_arr[np.argmax(max_arr)]  # pick rotation angle that aligns all rows;
        self.rot_angle = angle_opt
        self.image_rot = t.rotate(self.image, angle_opt)

        if plot:
            plt.figure()
            plt.plot(angle_arr, max_arr, ".", color="k")
            plt.plot(angle_arr, max_arr, color="k")
            plt.xlabel("Angle [deg]")

            plt.axvline(x=angle_opt, linestyle="--")
            plt.title(f"angle_opt = {angle_opt:.1f}deg")

        return angle_opt

    def find_peaks(self, nrows=3, ncols=10, min_distance=50):
        """
        obsolete, delete;
        """
        """
        Find the coordinates of the beams in a cross-section;
        Updates the list self.beam_coord_l;

        Parameters:
        - rows (int): number of expected rows (horizontal)
        - columns (int): number of expected columns (vertical)
        - min_distance: minimum distance between the center of the beams;        

        """

        if self.image_rot is not None:
            image = self.image_rot
            exp_num_peaks = nrows * ncols

            peak_arr = feature.peak_local_max(image, num_peaks=exp_num_peaks, min_distance=min_distance)
            self.beam_coord_l = peak_arr
        else:
            print("First evaluate rotation angle of cross section")
            peak_arr = None

        return peak_arr

    def find_geom(self):
        """
        Finds geometric properties of beam disposition:
            1. off_x
            2. off_y
            3. spacing
            4. rot_angle =
        """

        # procedure to find angle
        self.off_x = None
        self.off_y = None
        self.spacing = None

    def id_to_coord(id_x, id_y):
        """
        Converts the index of the beam to the respective coordinate
        e.g. first row, third column --> id_x = 1, id_y = 2

        Returns: coord_x, coord_y
        """
        coord_x = None
        coord_y = None
        return coord_x, coord_y

class Beam:
    def __init__(self, id_x, id_y):
        self.id_x = int(id_x)
        self.id_y = int(id_y)

        #initialised by tomo.init_coords()
        self.beam_coord_l = list()  # of centroid [x,y,z] at different z values --> updated by tomo.complete_beam_coords()
        self.beam_width_l = list() # sigma of gaussian fit --> radius at 1/e^2
        self.roi_l = list() #list of 2d arrays with ROIs of each beam;

        #
        self.div_full_angle = None

        # direction cosines
        self.e_x = None
        self.e_y = None
        self.e_z = None

    def __repr__(self):
        return f"Beam object, id_x = {self.id_x}, id_y = {self.id_y}"

    def find_coords(self, cross_section_l):
        n_layers = len(cross_section_l)

        for i in range(n_layers):
            print("ok")

    def find_dir_cos(self, px_size, debug = False):
        """
        Find direction cosines for the beam:
        e_x = v_x / |v|
        e_y = v_y / |v|
        e_z = v_z / |v|

        Find tilt of the beam with respect to z axis.
        """

        if self.max_z_idx_fit is None: #max_z_idx_fit limits the observations used for fitting the beam route.
            max_z_idx = len(self.beam_coord_l)
        else:
            max_z_idx = self.max_z_idx_fit

        # organise data
        x_arr = np.array(self.beam_coord_l)[:max_z_idx, 0]
        y_arr = np.array(self.beam_coord_l)[:max_z_idx, 1]
        z_arr = np.array(self.beam_coord_l)[:max_z_idx, 2] * (10 ** -3) / px_size  # z_arr in same units [px]
        t_arr = np.arange(0, len(x_arr))

        def pos(t, x0, v):
            return x0 + v * t

        # Fit curves
        popt, _ = curve_fit(pos, t_arr, x_arr)
        x0, v_x = popt
        popt, _ = curve_fit(pos, t_arr, y_arr)
        y0, v_y = popt
        popt, _ = curve_fit(pos, t_arr, z_arr)
        z0, v_z = popt

        v = np.array([v_x, v_y, v_z])
        e_x = v_x / np.linalg.norm(v)  # direction cosines
        e_y = v_y / np.linalg.norm(v)
        e_z = v_z / np.linalg.norm(v)

        self.e_x = e_x
        self.e_y = e_y
        self.e_z = e_z

        print(f"The direction cosines of the beam idx={self.id_x}, idy = {self.id_y} have been updated:")
        print(f"e_x = {self.e_x:.3f}, e_y = {self.e_y:.3f}, e_z = {self.e_z:.3f}")

        if debug:
            # Plot
            f, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 3))
            ax[0].plot(t_arr, x_arr, ".")
            ax[0].plot(t_arr, pos(t_arr, x0, v_x))
            ax[0].set_title(f"v_x = {v_x:.2f}")
            ax[0].set_xlabel("Step idx [int]")
            ax[0].set_ylabel("Position [px]")

            ax[1].plot(t_arr, y_arr, ".")
            ax[1].plot(t_arr, pos(t_arr, y0, v_y))
            ax[1].set_title(f"v_y = {v_y:.2f}")
            ax[1].set_xlabel("Step idx [int]")
            ax[1].set_ylabel("Position [px]")

            ax[2].plot(t_arr, z_arr, ".")
            ax[2].plot(t_arr, pos(t_arr, z0, v_z))
            ax[2].set_title(f"v_z = {v_z:.2f}")
            ax[2].set_ylabel("Position [px]")
            ax[2].set_xlabel("Step idx [int]")

            plt.tight_layout()

            # Print information
            alpha = np.degrees(np.arccos(e_x))
            beta = np.degrees(np.arccos(e_y))
            gamma = np.degrees(np.arccos(e_z))

            print(f"e_x = {e_x:.2f} --> alpha = {alpha:.2f}deg")
            print(f"e_y = {e_y:.2f} --> beta = {beta:.2f}deg")
            print(f"e_z = {e_z:.2f} --> gamma = {gamma:.2f}deg")

    def find_div(self):
        """
        Find beam full-angle divergence at 1/e^2
        """
        beam_coord_l = np.array(self.beam_coord_l)




    def plot_trajectory(self, limit_z_fit = False):
        """
        Plot the x-z and y-z diagrams of the beam trajectory

        :return: f, ax
        """
        beam_coord = np.array(self.beam_coord_l)

        if limit_z_fit: #excludes trajectory points beyond a certain z value.
            max_z_id = self.max_z_idx_fit
        else:
            max_z_id = len(self.beam_coord_l)

        x = beam_coord[:max_z_id, 0]
        y = beam_coord[:max_z_id, 1]
        z = beam_coord[:max_z_id, 2]

        f, ax = plt.subplots(ncols=2)
        ax[0].plot(x, z, ".")
        ax[0].set_title(f"({self.id_x}, {self.id_y}) x-z")
        ax[0].set_xlabel("X [px]")
        ax[0].set_ylabel("Z [mm]")

        ax[1].plot(y, z, ".")
        ax[1].set_title("y-z")
        ax[1].set_xlabel("Y [px]")
        ax[1].set_ylabel("Z [mm]")

        plt.tight_layout()

        return f, ax

    def plot_width(self, limit_z_fit = False):
        beam_coord = np.array(self.beam_coord_l)

        if limit_z_fit: #excludes trajectory points beyond a certain z value.
            try:
                max_z_id = self.max_z_idx_fit
            except AttributeError:
                print(f"Please define the max_z_id with tomo.set_max_z() to limit the fitting to valid cross sections.")
        else:
            max_z_id = len(self.beam_coord_l)

        z_arr = beam_coord[:max_z_id, 2]
        beam_width_l = np.array(self.beam_width_l)
        width_x_arr = beam_width_l[:max_z_id, 0]
        width_y_arr = beam_width_l[:max_z_id, 1]

        f, ax = plt.subplots()
        ax.plot(z_arr, width_x_arr, label = "x")
        ax.plot(z_arr, width_y_arr, label="y")
        ax.set_xlabel("Z[mm]")
        ax.set_ylabel("width [px]")
        ax.legend()

        plt.tight_layout()

        return f, ax

    def plot_rois(self):
        """
        Plots the regions of interest of the beam, at each cross section;

        :return:
        """
        roi_l = self.roi_l
        for i, roi_i in enumerate(roi_l):
            z_i = self.beam_coord_l[i][2]

            plt.figure()
            plt.imshow(roi_i)
            plt.title(f"z_idx = {i}, z = {z_i:.2f}mm")
            plt.xlabel("x [px]")
            plt.ylabel("y [px]")




"""
#Script
filenames = [item for item in os.listdir() if ".pkl" in item]

#Parameters
filename = filenames[4]
shape = (3, 10)
roi_width = 100

#load data
tomo = Tomography(filename, (3,10), roi_width)
tomo.load_data()
tomo.find_rot_spacing(angle_step = 1.0)
tomo.init_coord()
tomo.complete_coords(id_x = 0, id_y = 9)
"""
