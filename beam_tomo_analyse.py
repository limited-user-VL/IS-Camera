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
# standard library imports
import pickle
import numpy as np
import os
from datetime import datetime

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
from scipy.stats import norm


from matplotlib.patches import Rectangle

# 3rd party imports

# local library imports
def find_min_and_argmin(arr):
    """
    Returns the maximum value and the index of the maximum value in the array.

    Args:
    arr (numpy.ndarray): A NumPy array.

    Returns:
    tuple: (max_value, index_of_max_value)
    """
    min_value = np.min(arr)
    argmin_value = np.argmin(arr)
    return min_value, argmin_value

def normalise_arr(arr):
    """
    Normalises array arr, by subtracting minimum value and dividing by maximum

    return: arr
    """
    arr = np.array(arr)
    arr_ = arr-np.min(arr)
    norm_arr = arr_ / np.max(arr_)
    return norm_arr

def find_linear_function(point1, point2):
    """
    Returns the slope (m) and y-intercept (b) of the linear function passing through two points.

    Args:
    point1 (tuple): A tuple representing the first point (x1, y1).
    point2 (tuple): A tuple representing the second point (x2, y2).

    Returns:
    tuple: (m, b) where m is the slope and b is the y-intercept of the line y = mx + b.
    """
    x1, y1 = point1
    x2, y2 = point2

    # Calculate the slope
    m = (y2 - y1) / (x2 - x1)

    # Calculate the y-intercept
    b = y1 - m * x1

    return m, b

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

def gaussian(x, mean, standard_deviation):
    return 1 * np.exp( - (x - mean)**2 / (2 * standard_deviation ** 2))

def get_roi(image, center_x, center_y, roi_width,):
    """
    Given a 2D array, the function returns a region of interest within the 2D array, specified
    by the center coordinates (x,y) and the width of the region of interest in x and y;
    :param image: 2d numpy array
    :param center_x: int
    :param center_y: int
    :param roi_width: int
    :return: roi: 2d numpy array
    """
    roi = image[int(center_x - roi_width / 2.): int(center_x + roi_width / 2.),
          int(center_y - roi_width / 2.): int(center_y + roi_width / 2.)]
    return roi


class Tomography:
    def __init__(self, filename, shape, roi_width = 100):
        """
        Instantiate Tomography measurement.

        Args:
        1. Filename (str): File location and name
        2. Shape (tuple): (n_rows, n_cols), e.g (3,2)
        3. roi_width (int): width of region of interest in pixels

        """
        self.filename = filename
        self.directory = os.getcwd()
        self.shape = shape
        self.roi_width = int(roi_width)

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

        #mean direction cosine of beams
        self.mean_dir_cos = None # Updated by tomo.find_mean_dir_cos()

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
        opt_angle = cross_sect_i.find_rot(angle_min = angle_min, angle_max = angle_max,
                                          angle_step = angle_step, plot = False)
        print(f"Optimal rotation angle = {opt_angle:.2f}deg")

        # 2. Extract grid spacing;
        print("Extracting the grid spacing")
        cross_sect_i = self.cross_sect_l[0]
        peak_arr = cross_sect_i.find_peaks(exp_num_peaks = int(self.shape[0]*self.shape[1]), min_distance = 50)

        # label beams - line_id, col_id
        kmeans_rows = KMeans(n_clusters=self.shape[0])
        kmeans_rows.fit(peak_arr[:, 0].reshape(-1, 1))  # kmeans, 1 cluster per row
        coords_rows = kmeans_rows.cluster_centers_
        mean_delta_x = np.mean(np.diff(np.sort(coords_rows, axis=0), axis=0))  # spacing between rows
        spacing = mean_delta_x
        print(f"Average spacing [px] between beams = {spacing:.2f}")

        # 3. The rotation angle, rotated images and spacing are written to the cross-section objects
        print("Updating the rotation angle and rotated image for each cross section.")
        self.spacing_px = int(spacing)
        self.spacing_mm = float(spacing * self.pixel_size * 10**3)
        for cross_sect_i in tqdm(self.cross_sect_l):
            cross_sect_i.rot_angle = opt_angle
            cross_sect_i.image_rot = t.rotate(cross_sect_i.image, cross_sect_i.rot_angle, preserve_range=True)
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
        peak_arr = feature.peak_local_max(image_i, num_peaks=exp_num_peaks, min_distance=int(spacing * 0.5))
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
                beam_i = self.beam_l[id_x][id_y]
                beam_i.beam_coord_l = [[] for _ in range(self.n_sections)] #init beam.beam_coord_l
                beam_i.beam_coord_l[0] = peak_sorted_arr[id_x, id_y, 0]
                beam_i.beam_width_l = [[] for _ in range(self.n_sections)] #init beam.beam_coord_l
                beam_i.beam_intensity_l = [[] for _ in range(self.n_sections)] #init beam.beam_intensity_l

                beam_i.roi_l = [[] for _ in range(self.n_sections)] #initialise beam_i.roi_l
                beam_i.roi_fit_params_l = [[] for _ in range(self.n_sections)]
                beam_i.i_row_l = [[] for _ in range(self.n_sections)]
                beam_i.i_col_l = [[] for _ in range(self.n_sections)]

        print("Coordinates of beam in first layer were determined.")

    def find_coords_and_widths(self, debug = False):
        """
        Calls beam_i.complete_coords iteratively to cover all beams on the chip
        updates:
         1. beam_i.beam_coord_l
         2. beam_i.beam_width_l
         3. beam_i.roi_l

        :param debug:
        :return:
        """
        self.init_coords() #determines cords of the first cross section

        for id_x in tqdm(range(self.shape[0])):
            for id_y in range(self.shape[1]):
                beam_i = self.beam_l[id_x][id_y]
                beam_i.find_coords_and_widths(self.cross_sect_l, self.cross_sect_z_l, self.roi_width, debug = debug)

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

    def find_dir_cos(self, limit_z_fit = True, debug = False):
        """
        Iterates over beams and calls method of Beam class find_dir_cos.
        The direction cosine attributes of the Beam instances are determined.

        limit_z_fit [bool] - is the fit limited to the useful cross sections?

        :param debug:
        :return: None
        """

        if limit_z_fit:
            print("The fit is limited to the useful cross sections, defined by tomo.set_max_z()")

            if self.max_z_idx_fit == None:
                print("The useful cross sections have not been defined. Call tomo.set_max_z()")
                return None

        for id_x in range(self.shape[0]):
            for id_y in range(self.shape[1]):
                beam_i = self.beam_l[id_x][id_y]
                beam_i.find_dir_cos(self.pixel_size, limit_z_fit = limit_z_fit, debug = debug)

    def find_div(self, limit_z_fit = True, debug = False):
        """
        Find the divergence of all beams in the chip, by iterating over
        individual beams: beam_i.find_div().

        limit_z_fit = True - limits the fits in the coordinates and width calculations
        debug = True - produces extra prints and plots for debugging purposes

        """

        if limit_z_fit:
            print("The fit is limited to the useful cross sections, defined by tomo.set_max_z()")

            if self.max_z_idx_fit == None:
                print("The useful cross sections have not been defined. Call tomo.set_max_z()")
                return None

        for id_x in range(self.shape[0]):
            for id_y in range(self.shape[1]):
                beam_i = self.beam_l[id_x][id_y]
                beam_i.find_div(self.pixel_size, limit_z_fit = limit_z_fit, debug = debug)

    def find_mean_dir_cos(self):
        dir_cos_store = list()
        for id_x in range(self.shape[0]):
            for id_y in range(self.shape[1]):
                beam_i = self.beam_l[id_x][id_y]
                e_x, e_y, e_z = beam_i.e_x, beam_i.e_y, beam_i.e_z
                dir_cos_store.append([e_x, e_y, e_z])

        dir_cos_store = np.array(dir_cos_store)
        vec_mean = np.mean(dir_cos_store, axis=0)
        vec_mean = vec_mean / np.linalg.norm(vec_mean)

        self.mean_dir_cos = vec_mean

        print(f"Calculated average direction cosine [e_x, e_y, e_z] = [{vec_mean[0]:.2f}, {vec_mean[1]:.2f}, {vec_mean[2]:.2f}]")

        return vec_mean

    def find_dir_cos_div(self, limit_z_fit = True, debug = False):
        """
        Finds the direction cosines and divergence for each beam in the array.
        Finds the mean direction for all beams and write it to self.mean_dir_cos;

        Parameters:
            limit_z_fit [bool]: if True, the fit used for extracting the direction cosines
            and divergence is limited to some useful layers, whose z<tomo.max_z_fit:
            debug [bool]: if True, prints and plots to help debugging;
        """

        self.find_dir_cos(limit_z_fit = limit_z_fit, debug = debug)
        self.find_div(limit_z_fit = limit_z_fit, debug = debug)
        self.find_mean_dir_cos()

    def plot_dir_single(self, save = False):
        """
        Plots the direction of all beams in a single polar plot.
        #TODO Rotate reference frame such that the average direction coincides with
        # the z axis.

        """
        vec_mean = self.mean_dir_cos

        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'},
                               figsize=(5, 5),  # (2 * tomo.shape[1], 2 * tomo.shape[0]),
                               nrows=1,
                               ncols=1)

        theta_deg = np.degrees(np.arccos(vec_mean[2]))
        phi_rad = np.arctan(vec_mean[1] / vec_mean[0])
        print(f"phi_deg = {np.degrees(phi_rad):.2f}, theta_deg = {theta_deg:.2f}")
        ax.scatter(phi_rad, theta_deg, marker="o", s=120, color="black")
        ax.set_rmax(10)
        # ax.set_rlim(10)

        # plot direction cosines
        for id_x in range(self.shape[0]):
            for id_y in range(self.shape[1]):
                beam_i = self.beam_l[id_x][id_y]

                e_x, e_y, e_z = beam_i.e_x, beam_i.e_y, beam_i.e_z
                vec = [e_x, e_y, e_z]

                theta_deg = np.degrees(np.arccos(vec[2]))
                phi_rad = np.arctan(vec[1] / vec[0])
                ax.plot(phi_rad, theta_deg, ".", color="red")

        # plot direction cosines in rotated frame average #TODO
        beam_i = self.beam_l[id_x][id_y]

        vec = [beam_i.e_x, beam_i.e_y, beam_i.e_z]
        vec = vec - vec_mean
        vec = vec / np.linalg.norm(vec)
        #print(vec)
        theta_deg = np.degrees(np.arccos(vec[2]))
        phi_rad = np.arctan(vec[1] / vec[0])
        # ax.plot(phi_rad, theta_deg, ".", color = "green")
        # In the future apply transform to all points that maps the average vector to 0,0,1.

        if save:
            # Get current date and time
            now_datetime = datetime.now()
            datetime_string = now_datetime.strftime("%Y-%m-%d %H_%M")

            folder_name = "analysis"
            image_name = f"{datetime_string}_beam_directions_single.png"

            path_name = os.path.join(folder_name, image_name)
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)

            plt.savefig(path_name, dpi=300)
            print(f"Figure was saved in home directory as {image_name}.")

    def plot_div(self, save = False):
        """
        Plots 2 grids with the x-divergence and y-divergence of the beams plotted in an array of
        colormaps.

        Parameters:
            save [Bool] - save grid with color encoded divergence for x and y;
        """

        #generate grid points
        x_values = np.arange(0, self.shape[0])
        y_values = np.arange(1, self.shape[1])
        x, y = np.meshgrid(x_values, y_values, indexing="ij")

        div_x_arr = [[self.beam_l[id_x][id_y].div_x for id_y in range(x.shape[1])] for id_x in range(y.shape[0])]
        div_y_arr = [[self.beam_l[id_x][id_y].div_y for id_y in range(x.shape[1])] for id_x in range(y.shape[0])]

        # Plot using a colormap -div x
        f1, ax = plt.subplots(figsize=(8, 3))
        colormap = plt.pcolormesh(y, x, div_x_arr, cmap='viridis', shading='auto')
        for (i, j), val in np.ndenumerate(div_x_arr):
            plt.text(y_values[j], x_values[i], f"{val:.2f}", ha='center', va='center', color='white')

        plt.colorbar(ax = ax)
        ax.set_xlabel('y')
        ax.set_ylabel('x')
        ax.set_title('Div_x - full angle [deg]')
        plt.tight_layout()

        # Plot using a colormap -div y
        f2, ax = plt.subplots(figsize=(8, 3))
        colormap = plt.pcolormesh(y, x, div_y_arr, cmap='viridis', shading='auto')
        for (i, j), val in np.ndenumerate(div_y_arr):
            plt.text(y_values[j], x_values[i], f"{val:.2f}", ha='center', va='center', color='white')
        plt.colorbar(ax = ax)
        ax.set_xlabel('y')
        ax.set_ylabel('x')
        ax.set_title('Div_y - full angle [deg]')
        plt.tight_layout()

        if save:
            folder_name = "analysis"
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)

            # Get current date and time
            now_datetime = datetime.now()
            datetime_string = now_datetime.strftime("%Y-%m-%d %H_%M")
            image_name_1 = f"{datetime_string}_beam_div_x.png"
            path_name_1 = os.path.join(folder_name, image_name_1)
            f1.savefig(path_name_1, dpi = 300)
            print(f"Figure was saved as {path_name_1}.")

            image_name_2 = f"{datetime_string}_beam_div_y.png"
            path_name_2 = os.path.join(folder_name, image_name_2)
            f2.savefig(path_name_2, dpi=300)
            print(f"Figure was saved as {path_name_2}.")

    def plot_div_hist(self, save = False):
        """
        Plot a histogram with the divergence angles of the beams in x and y.
        """
        # generate grid points
        x_values = np.arange(0, self.shape[0])
        y_values = np.arange(1, self.shape[1])
        x, y = np.meshgrid(x_values, y_values, indexing="ij")

        div_x_arr = [[self.beam_l[id_x][id_y].div_x for id_y in range(x.shape[1])] for id_x in range(y.shape[0])]
        div_y_arr = [[self.beam_l[id_x][id_y].div_y for id_y in range(x.shape[1])] for id_x in range(y.shape[0])]
        div_x_arr = np.reshape(div_x_arr, (-1,))
        div_y_arr = np.reshape(div_y_arr, (-1,))

        #Plot
        f, ax = plt.subplots(ncols=2)

        # x Data
        # generate histogram - x
        hist_data_x, bins_x, _ = ax[0].hist(div_x_arr, bins=20, label="x", color="tab:blue", density=True)
        ax[0].set_title("Div x (full-angle)")
        ax[0].set_xlabel("(full-angle) [deg]")

        # fit gaussian curve to data
        mu, std = norm.fit(div_x_arr)
        xmin, xmax = ax[0].get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        ax[0].plot(x, p, 'k', linewidth=2, )
        ax[0].set_title(f"Div - x\nmu = {mu:.2f}°, std = {std:.2f}")

        # y Data
        # generate histogram - y
        hist_data_y, bins_y, _ = ax[1].hist(div_y_arr, bins=20, label="y", color="tab:blue", density=True)
        ax[1].set_title("Div y (full-angle)")
        ax[1].set_xlabel("(full-angle) [deg]")

        # fit gaussian curve to data
        mu, std = norm.fit(div_y_arr)
        xmin, xmax = ax[1].get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        ax[1].plot(x, p, 'k', linewidth=2, )
        ax[1].set_title(f"Div - y\nmu = {mu:.2f}°, std = {std:.2f}")

        if save:
            folder_name = "analysis"
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)

            # Get current date and time
            now_datetime = datetime.now()
            datetime_string = now_datetime.strftime("%Y-%m-%d %H_%M")
            image_name = f"{datetime_string}_beam_div_hist.png"
            path_name = os.path.join(folder_name, image_name)
            f.savefig(path_name, dpi = 300)
            print(f"Figure was saved as {path_name}.")

    def plot_dir(self, save = False):
        fig, ax_arr = plt.subplots(subplot_kw={'projection': 'polar'},
                                   figsize=(2 * self.shape[1], 2 * self.shape[0]),
                                   nrows=self.shape[0],
                                   ncols=self.shape[1])

        for id_x in range(self.shape[0]):
            for id_y in range(self.shape[1]):
                beam_i = self.beam_l[id_x][id_y]
                e_x, e_y, e_z = beam_i.e_x, beam_i.e_y, beam_i.e_z

                theta = np.arccos(e_z)
                phi = np.arctan(e_y / e_x)
                label = f"{id_x:.0f}x{id_y:.0f}, $\\theta = {{ {np.degrees(theta):.1f} }}^\circ, \\phi = {{ {np.degrees(phi):.1f} }}^\circ$"

                ax_arr[id_x][id_y].plot(phi, np.degrees(theta), "o")
                ax_arr[id_x][id_y].set_title(f"{label}", fontsize=10)
                ax_arr[id_x][id_y].set_rlim(0, 15)

        plt.tight_layout()

        if save:
            # Get current date and time
            now_datetime = datetime.now()
            datetime_string = now_datetime.strftime("%Y-%m-%d %H_%M")

            folder_name = "analysis"
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)

            image_name = f"{datetime_string}_beam_directions.png"
            path_name = os.path.join(folder_name, image_name)

            plt.savefig(path_name, dpi = 300)
            print(f"Figure was saved as {path_name}.")

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

    def plot_uniformity(self, save = True):
        """
        Plots a 2D grid, where each entry shows the maximum intensity value within the
        region of interest of each beam, in the lowest cross section;

        Returns handles of the figure and axis: f, ax
        """
        i_map = [[self.beam_l[id_x][id_y].beam_intensity_l[0] for id_y in range(self.shape[1])] for id_x in
                 range(self.shape[0])]

        f, ax = plt.subplots()
        im = ax.imshow(i_map, origin="lower")
        ax.set_xlabel("Column #")
        ax.set_ylabel("Row #")
        plt.colorbar(im, ax = ax)
        ax.set_title(f"Min/Max = {np.min(i_map) / np.max(i_map):.2f}")

        if save:
            # Get current date and time
            now_datetime = datetime.now()
            datetime_string = now_datetime.strftime("%Y-%m-%d %H_%M")

            folder_name = "analysis"
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)

            image_name = f"{datetime_string}_uniformity.png"
            path_name = os.path.join(folder_name, image_name)

            plt.savefig(path_name, dpi=300)
            print(f"Figure was saved as {path_name}.")
            
        return f, ax


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
        self.image_rot = t.rotate(self.image, angle_opt, preserve_range=True)

        if plot:
            plt.figure()
            plt.plot(angle_arr, max_arr, ".", color="k")
            plt.plot(angle_arr, max_arr, color="k")
            plt.xlabel("Angle [deg]")

            plt.axvline(x=angle_opt, linestyle="--")
            plt.title(f"angle_opt = {angle_opt:.1f}deg")

        return angle_opt

    def find_peaks(self, exp_num_peaks = 50, min_distance=50):
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
        self.beam_coord_l = list() # of centroid [x,y,z] at different z values --> updated by tomo.complete_beam_coords()
        self.beam_width_l = list() # sigma of gaussian fit --> radius at 1/e^2
        self.beam_intensity_l = list() #updated by beam.find_coords_and_widths()

        self.roi_l = list() #list of 2d arrays with ROIs of each beam;
        self.roi_fit_params_l = list() #arr is initiaed by tomo.init_coords() and filled by tomo.complete_beam_coords()
        self.i_row_l = list() #stores I(x) within ROI, for each cross section of the beam
        self.i_col_l = list() #stores I(y) within ROI, ...
                           #[col_mu, row_mu, col_sigma, row_sigma]
        #
        self.div_full_angle = None

        # direction cosines
        self.e_x = None #updated by beam_i.find_dir_cos()
        self.e_y = None
        self.e_z = None

        # divergence (deg)
        self.div_x = None #updated by beam_i.find_div()
        self.div_y = None

    def __repr__(self):
        return f"Beam object, id_x = {self.id_x}, id_y = {self.id_y}"

    def find_dir_cos(self,  px_size, limit_z_fit = True, debug = False):
        """
        Find direction cosines for the beam:
        e_x = v_x / |v|
        e_y = v_y / |v|
        e_z = v_z / |v|

        Find tilt of the beam with respect to z axis.
        """

        if limit_z_fit:  # excludes trajectory points beyond a certain z value.
            try:
                max_z_idx = self.max_z_idx_fit
            except AttributeError:
                print(f"Please define the max_z_id with tomo.set_max_z() to limit the fitting to valid cross sections.")
        else:
            max_z_idx = len(self.beam_coord_l)

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
            ax[0].set_title(f"v_x = {v_x:.2f}, e_x = {e_x:.2f}")
            ax[0].set_xlabel("Step idx [int]")
            ax[0].set_ylabel("Position [px]")

            ax[1].plot(t_arr, y_arr, ".")
            ax[1].plot(t_arr, pos(t_arr, y0, v_y))
            ax[1].set_title(f"v_y = {v_y:.2f}, e_y = {e_y:.2f}")
            ax[1].set_xlabel("Step idx [int]")
            ax[1].set_ylabel("Position [px]")

            ax[2].plot(t_arr, z_arr, ".")
            ax[2].plot(t_arr, pos(t_arr, z0, v_z))
            ax[2].set_title(f"v_z = {v_z:.2f}, e_z = {e_z:.2f}")
            ax[2].set_ylabel("Position [px]")
            ax[2].set_xlabel("Step idx [int]")
            plt.tight_layout()

            #Save image to debug_folder
            folder_name = "debug"
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)

            image_name = f"beam_x{self.id_x}_y{self.id_y}_dir_cos_plot.png"
            path_name = os.path.join(folder_name, image_name)
            plt.savefig(path_name, dpi = 300)

            # Print information
            alpha = np.degrees(np.arccos(e_x))
            beta = np.degrees(np.arccos(e_y))
            gamma = np.degrees(np.arccos(e_z))

            print(f"e_x = {e_x:.2f} --> alpha = {alpha:.2f}deg")
            print(f"e_y = {e_y:.2f} --> beta = {beta:.2f}deg")
            print(f"e_z = {e_z:.2f} --> gamma = {gamma:.2f}deg")

    def find_div(self, pixel_size, limit_z_fit = True, debug = False):
        """
        Find beam full-angle divergence at 1/e^2
        """
        beam_coord_l = np.array(self.beam_coord_l)
        # tomo.find_div(debug = True) #loops over beam method beam_i.find_div
        beam_coord = np.array(self.beam_coord_l)

        if limit_z_fit:  # excludes trajectory points beyond a certain z value.
            try:
                max_z_id = self.max_z_idx_fit
            except AttributeError:
                print(f"Please define the max_z_id with tomo.set_max_z() to limit the fitting to valid cross sections.")
        else:
            max_z_id = len(self.beam_coord_l)

        z_arr = beam_coord[:max_z_id, 2]
        z_px_arr = (z_arr * 10 ** -3) / pixel_size
        beam_width_l = np.array(self.beam_width_l)
        width_y_arr = beam_width_l[:max_z_id, 0]
        width_x_arr = beam_width_l[:max_z_id, 1]

        # y - fit
        linear_f = lambda x, m, b: m * x + b
        popt, _ = curve_fit(linear_f, z_px_arr, width_y_arr, )
        width_y_fit_arr = linear_f(z_px_arr, *popt)
        m_y = popt[0]
        theta_y = np.degrees(np.arctan(m_y))

        # x - fit - sum over col
        popt, _ = curve_fit(linear_f, z_px_arr, width_x_arr, )
        width_x_fit_arr = linear_f(z_px_arr, *popt)
        m_x = popt[0]
        theta_x = np.degrees(np.arctan(m_x))

        # Update beam_i.attributes
        self.div_x = 2*theta_x #full angle = 2 * half-angle
        self.div_y = 2*theta_y

        # Plot
        if debug:
            print(f"The divergence angles of the beam idx={self.id_x}, idy = {self.id_y} have been updated:")
            print(f"div_x = {self.div_x:.3f}deg, e_y = {self.div_y:.3f}deg")

            f, ax = plt.subplots()
            plt.title(
                f"Debug - find_div,\n m_x = {m_x:.2f}, 2 x theta_x = {2*theta_x:.2f}deg ,\nm_y = {m_y:.2f}, 2 x theta_y = {2*theta_y:.2f}deg,")
            # ax.plot(z_px_arr, width_x_arr, label = "x")
            ax.plot(z_px_arr, width_y_arr, ".", label="y", color = "tab:blue")
            ax.plot(z_px_arr, width_y_fit_arr, label="y-fit", color = "tab:blue")

            ax.plot(z_px_arr, width_x_arr, ".", label="x", color = "tab:orange")
            ax.plot(z_px_arr, width_x_fit_arr, label="x-fit", color = "tab:orange")
            ax.set_xlabel("Z[px]")
            ax.set_ylabel("width [px]")
            ax.legend()
            plt.tight_layout()

            #Save image to debug_folder
            folder_name = "debug"
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)

            image_name = f"beam_x{self.id_x}_y{self.id_y}_div_plot.png"
            path_name = os.path.join(folder_name, image_name)
            plt.savefig(path_name, dpi = 300)

    def find_lin_backg(self, arr, debug=True):
        """
        Finds linear function mx+b that represents background.

        return: m, b
        """

        min_val_m, min_idx_m = find_min_and_argmin(arr[:int(len(arr) / 2)])
        min_val_p, min_idx_p = find_min_and_argmin(arr[int(len(arr) / 2):])

        p1 = (min_idx_m, min_val_m)
        p2 = (min_idx_p + int(len(arr) / 2), min_val_p)
        if debug:
            print("p1: ", p1)
            print("p2: ", p2)

        m, b = find_linear_function(p1, p2)
        return m, b

    def subtract_lin_backg(self,arr, debug=False):
        """
        1. Extract background, represented by linear function
        2. Subtract linear background from arr

        """
        m, b = self.find_lin_backg(arr, debug=debug)
        idx_arr = range(len(arr))
        backg_arr = m * idx_arr + b

        return arr - backg_arr

    def get_gaussian_specs(self, roi_img, roi_width, debug=False):
        """
        Given a 2D array with a region of interest (ROI) around a beam,
        fit a gaussian beam to the marginal X and marginal Y distributions
        and retrieve the middle point for X and Y and the sigma of the
        gaussian fit..

        :param roi_img: (2D array) - region of interest around beam
        :param roi_width (int) - width of region of interest in pixel units;
        :return: i_row, i_col, row_mu, col_mu, row_sigma, col_sigma
        """

        # Project intensity map of ROI into x and y --> I(x) = roi_sum_col, I(y) = roi_sum_row
        roi_sum_row_arr = np.sum(roi_img, axis=0)
        roi_sum_col_arr = np.sum(roi_img, axis=1)

        #if debug:
        #    plt.figure()
        #    plt.title("Debug beam_i.get_gaussian_specs- ROI of beam")
        #   plt.imshow(roi_img, origin = "lower")

        #get x arr for plotting
        row_idx_arr, col_idx_arr = range(len(roi_sum_row_arr)), range(len(roi_sum_col_arr))

        # Normalise I(x) and I(y)
        roi_sum_row_arr = normalise_arr(roi_sum_row_arr)
        roi_sum_col_arr = normalise_arr(roi_sum_col_arr)

        # Correct background - background has a slope --> mx+b --> subtract background(x) from I(x)
        roi_sum_row_arr = self.subtract_lin_backg(roi_sum_row_arr, debug=debug)
        roi_sum_row_arr = normalise_arr(roi_sum_row_arr)

        # m_col, b_col = find_backg_function(roi_sum_col_arr, debug = True)
        roi_sum_col_arr = self.subtract_lin_backg(roi_sum_col_arr, debug=debug)
        roi_sum_col_arr = normalise_arr(roi_sum_col_arr)

        i_row = roi_sum_row_arr #sum along cols
        i_col = roi_sum_col_arr

        try:
            mean_0 = roi_width / 2
            std_0 = roi_width / 5
            p0 = [mean_0, std_0]

            # fit sum of rows
            popt_row, _ = curve_fit(gaussian, row_idx_arr, i_row, p0=p0)
            row_mu = int(popt_row[0])
            row_sigma = int(popt_row[1])

            # fit sum of cols
            popt_col, _ = curve_fit(gaussian, col_idx_arr, i_col, p0=p0)
            col_mu = int(popt_col[0])
            col_sigma = int(popt_col[1])

            # sigma as average of sigma_x and sigma_y
            sigma = 0.5 * (row_sigma + col_sigma)

        except (RuntimeError, ValueError):  # if the fit fails
            row_mean = int(roi_width / 2)
            col_mean = int(roi_width / 2)
            popt_row = [mean_0, std_0]
            popt_col = [mean_0, std_0]
            sigma = int(roi_width / 5)

        if debug:
            plt.figure()
            plt.title("Debug - beam_i.get_gaussian_specs()")
            plt.plot(row_idx_arr, roi_sum_row_arr, ".", color="k")
            y_fit = gaussian(row_idx_arr, *popt_row)
            plt.plot(row_idx_arr, y_fit, label="row")
            plt.xlabel("[px]")

            plt.plot(col_idx_arr, roi_sum_col_arr, ".", color="k")
            y_fit = gaussian(row_idx_arr, *popt_col)
            plt.plot(row_idx_arr, y_fit, label="col")
            plt.legend()

        #return row_mu, col_mu, row_sigma, col_sigma
        return i_row, i_col, row_mu, col_mu, row_sigma, col_sigma

    def plot_trajectory(self, limit_z_fit = True):
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

    def plot_width(self, limit_z_fit = True):
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

    def plot_gauss_fit(self, limit_z_fit = False):
        """
        Plot for each cross section of the beam, the ROI, the intensity profiles in x and y and
        respective gaussian fits.

        """
        if limit_z_fit: #excludes trajectory points beyond a certain z value.
            try:
                max_id_z = self.max_z_idx_fit
            except AttributeError:
                print(f"Please define the max_z_id with tomo.set_max_z() to limit the fitting to valid cross sections.")
        else:
            max_id_z = len(self.beam_coord_l)

        for id_z in range(max_id_z):
            roi_i = self.roi_l[id_z]
            col_mu, row_mu, col_sigma, row_sigma = self.roi_fit_params_l[id_z]  # fit params
            i_col_arr = self.i_col_l[id_z]  # sum over columns
            i_row_arr = self.i_row_l[id_z]  # sum over lines
            idx_arr = np.arange(len(i_col_arr))

            # Plots
            f, ax = plt.subplots(ncols=2, nrows=1, figsize=(8, 4))
            ax[0].imshow(roi_i, origin="lower")
            ax[0].set_xlabel("Y [px]")
            ax[0].set_ylabel("X [px]")
            ax[0].set_title(f"id_x = {self.id_x},  id_y = {self.id_y}, id_z = {id_z}")

            # plot sum_col - x
            ax[1].plot(idx_arr, i_col_arr, ".", label="sum_col - x", color="tab:blue")
            col_fit = gaussian(idx_arr, col_mu, col_sigma)
            ax[1].plot(idx_arr, col_fit, color="tab:blue")

            # plot sum_col - y
            ax[1].plot(idx_arr, i_row_arr, ".", label="sum_row - y", color="tab:orange")
            row_fit = gaussian(idx_arr, row_mu, row_sigma)
            ax[1].plot(idx_arr, row_fit, color="tab:orange")

            ax[1].set_title(
                f"$\mu_x = {{{col_mu:.1f}}},  \sigma_x = {{{col_sigma:.1f}}}$ \n $\mu_y = {{{row_mu:.1f}}},  \sigma_y = {{{row_sigma:.1f}}}$")
            ax[1].set_xlabel("Pos [px]")
            ax[1].legend()

    def find_coords_and_widths(self, cross_sect_l, cross_sect_z_l, roi_width, debug = False):
        """
        Extracts the coordinates and width of the beam, by analysing its trajectory across
        several cross-sections.

        Requires cross_sect_l from Tomo instance;
        Requires roi_width

        """
        n_sections = len(self.beam_coord_l)

        for id_z in range(n_sections):
            #define initial_guess for center of ROI (region of interest)
            if id_z == 0:
                coord_x, coord_y, coord_z = self.beam_coord_l[id_z]
            else:
                coord_x, coord_y, coord_z = self.beam_coord_l[id_z-1]

            #get beam roi, based on coordinates of beam in lower section
            image_i = cross_sect_l[id_z].image_rot
            roi_i = get_roi(image_i, coord_x, coord_y,
                            roi_width = roi_width)

            roi_i = gaussian_filter(roi_i, sigma = roi_width / 10)  # smoothen ROI for pixel noise

            # establish new coordinates of beam in current z_id, with gaussian fit
            #row_mu corresponds to a sum over rows --> I(y)
            #col_mu corrsponds to a sum over cols --> I(x)
            i_row, i_col, row_mu, col_mu, row_sigma, col_sigma = self.get_gaussian_specs(roi_i, roi_width, debug=debug)
            coord_x = int(coord_x + (col_mu - roi_width / 2)) #col_mu --> summed over colums --> x
            coord_y = int(coord_y + (row_mu - roi_width / 2))
            coord_z = cross_sect_z_l[id_z]

            if debug:
                plt.figure()
                plt.title(f"Debug - beam_i.complete_coords_widths,\nz= {coord_z:.2f}mm")
                plt.imshow(roi_i, origin="lower")

            # update beam.roi_l and beam.beam_coord_l
            self.beam_coord_l[id_z] = np.array([coord_x, coord_y, coord_z])
            self.beam_width_l[id_z] = np.array([row_sigma, col_sigma]) #row_sigma --> sum over rows --> I(y)
            self.beam_intensity_l[id_z] = np.max(roi_i)

            self.roi_l[id_z] = roi_i
            self.roi_fit_params_l[id_z] = [col_mu, row_mu, col_sigma, row_sigma]
            self.i_row_l[id_z] = i_row
            self.i_col_l[id_z] = i_col

        if debug:
            print(f"The coordinates of beam ({self.id_x:.0f},{self.id_y:.0f}) have been determined for all cross-sections.")


"""
#Script
# Access folder with data in G drive
folder_path = "G:\Shared drives\VitreaLab Share\Lab Data\Light Engine Team\X-Reality Projects (XR)\Augmented Reality (AR)\Lab Data\AR\2023-12-05\beam_tomography\\"
folder_path = r"G:\Shared drives\VitreaLab Share\Lab Data\Light Engine Team\X-Reality Projects (XR)\Augmented Reality (AR)\Lab Data\AR\2023-12-05\beam_tomography\\"
files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
filenames = [item for item in files if ".pkl" in item]

# Parameters
filename = filenames[4]
file_path = os.path.join(folder_path, filename)
shape = (3, 10)
roi_width = 100

# Load data
tomo = Tomography(file_path, (3,10), roi_width)
tomo.load_data() #loads images into tomography object

# Extract beam coords and widths
tomo.find_rot_spacing(angle_min = 46, angle_max = 47, angle_step = 0.25)
tomo.find_coords_and_widths(debug = False) #Find coords for all the other layers

#Extract parameters, direction cosines and divergence for each beam
tomo.set_max_z(16.0) #limits fits to cross section with z<max_z;
tomo.find_dir_cos_div(limit_z_fit = True, debug = False)

#Visualise data
tomo.plot_div() #plots array with divergences in x and y
tomo.plot_dir() #plots the direction of each beam in an array of polar plots
tomo.plot_dir_single() #single plot for all beams

"""
