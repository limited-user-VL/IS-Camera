"""
This project is used to evaluate the output of Vitrealab light chips.
The goal is to extract the tilt angles (tilt_row, tilt_col) and beam
divergence (beam_div_row, beam_div_col) for each beam in the beam array.
!Note that variation in x corresponds to xx_row;
!Note that variation in y corresponds to xx_col, where I decided to stick to the "matrix convention" in the
 representation of the images, as it is done by plt.imshow(), for consistency.
x corresponds to the first index of the arrays and y the second index.

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

def convert_uint16_to_uint8(img_uint16):
    """
    Convert image encoded in 16bit integers to 8bit integers.
    To be used in generating videos from sequence of images
    """

    # Normalize the uint16 image to 0-1 range
    normalized = img_uint16 / np.max(img_uint16)

    # Scale to 0-255 and convert to uint8
    img_uint8 = (normalized * 255).astype(np.uint8)
    return img_uint8

class Tomography:
    def __init__(self, filename, shape, method = "stats", roi_width = 100, sub_backg = False):
        """
        Instantiate Tomography measurement.

        :param filename: File location and name
        :type filename: str

        :param shape: Tuple indicating hte shape of the grid of beams (n_rows, n_cols)
        :type shape: tuple


        :param method: "fit" or "stats". "fits" uses a gaussian fit to retrieve the position and width of each beam
        "stats" uses the normalised intensity distribution as a pdf to calculate the mean and sigma of the beam.

        :param roi_width: width of the region of interest in pixels
        :type roi_width: integer

        :param sub_backg: if true, tomo.find_pos_and_widths will subtract the background noise level (with linear
        function) before establishing the value of mu and sigma (position and width) of the beas.

        """
        self.filename = filename
        self.directory = os.getcwd()
        self.shape = shape
        self.roi_width = int(roi_width)
        self.sub_backg = sub_backg #if True

        if method == "fit" or method == "stats":
            self.method = str(method)  # define method used to extract mu and sigma of each beam.
            # method = "fit" or "stats"
        else:
            raise ValueError("Cannot initialise Tomography object. Please define method as fit or stats.")


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
        spacing = int(mean_delta_x)
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
        Finds the coordinates of the beams on the lowest cross-section and initialises the beams (Beam class) attributes
        The coordinates are then pushed into:
         1. CrossSection.beam_coord_l
         2. Beam_i.beam_coord_l

        :return:
        """
        # TODO: Function to generate movie for a single beam.

        # call tomo.coord_init()
        exp_num_peaks = self.shape[0] * self.shape[1]
        cross_0 = self.cross_sect_l[0]
        spacing = int(cross_0.spacing_px)
        image_i = self.cross_sect_l[0].image_rot

        #find peaks
        peak_arr = feature.peak_local_max(image_i,
                                          num_peaks=exp_num_peaks,
                                          min_distance=int(spacing * 0.5),
                                          threshold_rel=0.02)

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

                    d = np.sum(np.abs(peak_arr - np.array([coord_x, coord_y])), axis=1) #distance array
                    idx_min = np.argmin(d, axis=0) #point with smallest distance
                    if d[idx_min] < 0.6*spacing:
                        coord_i = np.concatenate((peak_arr[idx_min], np.array(self.cross_sect_z_l[0]).reshape(-1)),axis=0)
                        peak_sorted_arr[id_x][id_y][k] = coord_i
                    else: #Do not attribute a beam, if distance is larger than grid spacing
                        peak_sorted_arr[id_x][id_y][k] = None

        peak_sorted_arr = np.array(peak_sorted_arr)

        # pass peak coords to cross section objects
        cross_0.beam_coord_l = peak_sorted_arr

        # initialise beam attributes and pass peak coords into each beam, listed in self.beam_l
        for id_x in range(self.shape[0]):
            for id_y in range(self.shape[1]):
                beam_i = self.beam_l[id_x][id_y]
                beam_i.beam_coord_l = [[] for _ in range(self.n_sections)] #init beam.beam_coord_l
                beam_i.beam_coord_l[0] = peak_sorted_arr[id_x, id_y, 0]
                beam_i.beam_width_l = [[] for _ in range(self.n_sections)] #init beam.beam_coord_l
                beam_i.beam_intensity_l = [None for _ in range(self.n_sections)] #init beam.beam_intensity_l

                beam_i.roi_l = [None for _ in range(self.n_sections)] #initialise beam_i.roi_l
                beam_i.roi_beam_param_l = [None for _ in range(self.n_sections)]
                beam_i.i_row_l = [None for _ in range(self.n_sections)]
                beam_i.i_col_l = [None for _ in range(self.n_sections)]

        print("Coordinates of beam in first layer were determined.")

    def find_pos_and_widths(self, debug = False):
        """
        Calls beam_i.complete_coords iteratively to cover all beams on the chip
        updates:
         1. beam_i.beam_coord_l
         2. beam_i.beam_width_l
         3. beam_i.roi_l

        :param fit: True if the beam positions and widths are extracted with a gaussian fit method
        and False is a statistical method is used.
        :type fit: bool
        :return:
        """
        self.init_coords() #determines coords of the first cross section

        for id_x in tqdm(range(self.shape[0])):
            for id_y in range(self.shape[1]):
                beam_i = self.beam_l[id_x][id_y]
                beam_i.find_pos_and_widths(self.cross_sect_l, self.cross_sect_z_l, self.roi_width, method = self.method, sub_backg = self.sub_backg, debug = debug)
        print("Coordinates of the beams and respective widths have been determined in all cross sections")

    def set_max_z(self, max_z):
        """
        Establish max z-value [mm] of cross sections to be used in fitting to extract tilt_row, tilt_col, div_row, div_col
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
                beam_i.find_dir_cos(self.pixel_size, self.method, limit_z_fit = limit_z_fit, debug = debug)

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
                beam_i.find_div(self.pixel_size, self.method, limit_z_fit = limit_z_fit, debug = debug)

    def find_mean_dir_cos(self):
        dir_cos_store = list()
        for id_x in range(self.shape[0]):
            for id_y in range(self.shape[1]):
                beam_i = self.beam_l[id_x][id_y]
                e_x, e_y, e_z = beam_i.e_x, beam_i.e_y, beam_i.e_z
                if e_x != None: #for the case, where no beam is assigned to the grid point id_x, id_y
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

        self.find_dir_cos(limit_z_fit = limit_z_fit, debug = debug) #find direction cosines
        self.find_div(limit_z_fit = limit_z_fit, debug = debug) #find divergence
        self.find_mean_dir_cos() #find mean direction cosine

    def plot_dir_single(self, save = False):
        """
        Plots the direction of all beams in a single polar plot.
        # TODO Rotate reference frame such that the average direction coincides with
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
        fig.suptitle(f"Plot of beam direction. Method = {self.method}")

        # ax.set_rlim(10)

        # plot direction cosines
        for id_x in range(self.shape[0]):
            for id_y in range(self.shape[1]):
                beam_i = self.beam_l[id_x][id_y]
                e_x, e_y, e_z = beam_i.e_x, beam_i.e_y, beam_i.e_z
                vec = [e_x, e_y, e_z]

                if e_x != None:
                    theta_deg = np.degrees(np.arccos(vec[2]))
                    phi_rad = np.arctan(vec[1] / vec[0])
                    ax.plot(phi_rad, theta_deg, ".", color="red")
                else: #if grid point was not assigned a beam
                    continue

        plt.tight_layout()

        # TODO plot direction cosines in rotated frame average
        #beam_i = self.beam_l[id_x][id_y]
        #vec = [beam_i.e_x, beam_i.e_y, beam_i.e_z]
        #vec = vec - vec_mean
        #vec = vec / np.linalg.norm(vec)
        #print(vec)
        #theta_deg = np.degrees(np.arccos(vec[2]))
        #phi_rad = np.arctan(vec[1] / vec[0])
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
        y_values = np.arange(0, self.shape[1])
        x, y = np.meshgrid(x_values, y_values, indexing="ij")

        div_row_arr = [[self.beam_l[id_x][id_y].div_row
                        if self.beam_l[id_x][id_y].div_row!=None else -1 #None entries are converted to -1
                        for id_y in range(x.shape[1])]
                        for id_x in range(y.shape[0])]

        div_col_arr = [[self.beam_l[id_x][id_y].div_col
                        if self.beam_l[id_x][id_y].div_col !=None else -1
                        for id_y in range(x.shape[1])]
                       for id_x in range(y.shape[0])]

        # Plot using a colormap -div x
        f1, ax = plt.subplots(figsize=(8, 3))
        colormap = plt.pcolormesh(y, x, div_row_arr, cmap='viridis', shading='auto')
        for (i, j), val in np.ndenumerate(div_row_arr):
            plt.text(y_values[j], x_values[i], f"{val:.2f}", ha='center', va='center', color='white')

        plt.colorbar(ax = ax)
        ax.set_xlabel('y')
        ax.set_ylabel('x')
        ax.set_title(f'Div_row - full angle [deg], method = {self.method}')
        plt.tight_layout()

        # Plot using a colormap -div y
        f2, ax = plt.subplots(figsize=(8, 3))
        colormap = plt.pcolormesh(y, x, div_col_arr, cmap='viridis', shading='auto')
        for (i, j), val in np.ndenumerate(div_col_arr):
            plt.text(y_values[j], x_values[i], f"{val:.2f}", ha='center', va='center', color='white')
        plt.colorbar(ax = ax)
        ax.set_xlabel('y')
        ax.set_ylabel('x')
        ax.set_title(f'Div_col - full angle [deg], method = {self.method}')
        plt.tight_layout()

        if save:
            folder_name = "analysis"
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)

            # Get current date and time
            now_datetime = datetime.now()
            datetime_string = now_datetime.strftime("%Y-%m-%d %H_%M")
            image_name_1 = f"{datetime_string}_beam_div_row.png"
            path_name_1 = os.path.join(folder_name, image_name_1)
            f1.savefig(path_name_1, dpi = 300)
            print(f"Figure was saved as {path_name_1}.")

            image_name_2 = f"{datetime_string}_beam_div_col.png"
            path_name_2 = os.path.join(folder_name, image_name_2)
            f2.savefig(path_name_2, dpi=300)
            print(f"Figure was saved as {path_name_2}.")

    def plot_div_hist(self, save = False):
        """
        Plot a histogram with the divergence angles of the beams in x and y.
        """
        # generate grid points
        x_values = np.arange(0, self.shape[0])
        y_values = np.arange(0, self.shape[1])
        x, y = np.meshgrid(x_values, y_values, indexing="ij")

        div_row_arr = [[self.beam_l[id_x][id_y].div_row for id_y in range(x.shape[1])] for id_x in range(y.shape[0])]
        div_col_arr = [[self.beam_l[id_x][id_y].div_col for id_y in range(x.shape[1])] for id_x in range(y.shape[0])]
        div_row_arr = np.reshape(div_row_arr, (-1,))
        div_col_arr = np.reshape(div_col_arr, (-1,))

        #Remove None values from grid points without a beam
        div_row_arr = div_row_arr[div_row_arr != np.array(None)].astype(float)
        div_col_arr = div_col_arr[div_col_arr != np.array(None)].astype(float)

        #Plot
        f, ax = plt.subplots(ncols=2)

        # x Data
        # generate histogram - x
        hist_data_x, bins_x, _ = ax[0].hist(div_row_arr, bins=20, label="x", color="tab:blue", density=True)
        ax[0].set_xlabel("(full-angle) [deg]")

        # fit gaussian curve to data
        mu, std = norm.fit(div_row_arr)
        xmin, xmax = ax[0].get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        ax[0].plot(x, p, 'k', linewidth=2, )
        ax[0].set_title(f"Div - row\nmu = {mu:.2f}°, std = {std:.2f}\nmethod = {self.method}")

        # y Data
        # generate histogram - y
        hist_data_y, bins_y, _ = ax[1].hist(div_col_arr, bins=20, label="y", color="tab:blue", density=True)

        # fit gaussian curve to data
        mu, std = norm.fit(div_col_arr)
        xmin, xmax = ax[1].get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        ax[1].plot(x, p, 'k', linewidth=2, )
        ax[1].set_title(f"Div - col\nmu = {mu:.2f}°, std = {std:.2f}\nmethod = {self.method}")
        ax[1].set_xlabel("(full-angle) [deg]")

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

                if e_x != None: #if grid point id_x, id_y was assigned a beam
                    theta = np.arccos(e_z)
                    phi = np.arctan(e_y / e_x)
                    label = f"{id_x:.0f}x{id_y:.0f}, $\\theta = {{ {np.degrees(theta):.1f} }}^\circ, \\phi = {{ {np.degrees(phi):.1f} }}^\circ$"

                    ax_arr[id_x][id_y].plot(phi, np.degrees(theta), "o")
                    ax_arr[id_x][id_y].set_title(f"{label}", fontsize=10)
                    ax_arr[id_x][id_y].set_rlim(0, 15)
                else: #no beam assigned to grid point.
                    continue

        fig.suptitle(f"Method = {self.method}")
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

    def plot_cross_section(self, id_z, save = False):
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
                try:
                    row, col, z = rois_in_cross_section[id_x][id_y]

                    # Create a rectangle patch - !For the Rectangle obj, x is the horizontal axis;
                    rect = Rectangle((int(col-roi_width/2), int(row - roi_width/2)),
                                             roi_width, roi_width, linewidth=2, edgecolor='r', facecolor='none')
                    # Add the rectangle to the Axes
                    ax.add_patch(rect)
                except (ValueError, TypeError): #if row, col, z = None (no beam assigned to this grid point)
                    continue
        plt.title(f"id_z = {id_z}, z = {self.cross_sect_z_l[id_z]:.2f}mm")
        plt.tight_layout()

        if save:
            # Get current date and time
            now_datetime = datetime.now()
            datetime_string = now_datetime.strftime("%Y-%m-%d %H_%M")

            folder_name = "analysis"
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)

            image_name = f"{datetime_string}_cross_section_id_z_{id_z}.png"
            path_name = os.path.join(folder_name, image_name)

            plt.savefig(path_name, dpi=300)
            print(f"Figure was saved as {path_name}.")


        return f, ax

    def plot_uniformity(self, save = True):
        """
        Plots a 2D grid, where each entry shows the maximum intensity value within the
        region of interest of each beam, in the lowest cross section;

        Returns handles of the figure and axis: f, ax
        """
        i_map = [[self.beam_l[id_x][id_y].beam_intensity_l[0]
                  if self.beam_l[id_x][id_y].beam_intensity_l[0] != None
                  else -1
                  for id_y in range(self.shape[1])]
                 for id_x in range(self.shape[0])]

        i_arr = np.array(i_map).reshape(-1)
        i_arr = i_arr[i_arr != -1]  # exclude negative points

        f, ax = plt.subplots()
        im = ax.imshow(i_map, origin="lower")
        ax.set_xlabel("Column #")
        ax.set_ylabel("Row #")
        plt.colorbar(im, ax = ax)
        ax.set_title(f"Chip Uniformity\nMin/Max = {np.min(i_arr) / np.max(i_arr):.2f}")

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

    def generate_beam_movie(self, id_x, id_y):
        beam_i = self.beam_l[id_x][id_y]



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
        self.beam_intensity_l = list() #updated by beam.find_pos_and_widths()

        self.roi_l = list() #list of 2d arrays with ROIs of each beam;
        self.roi_beam_param_l = list() #arr is initiaed by tomo.init_coords() and filled by tomo.complete_beam_coords()
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
        self.div_row = None #updated by beam_i.find_div()
        self.div_col = None

    def __repr__(self):
        return f"Beam object, id_x = {self.id_x}, id_y = {self.id_y}"

    def find_dir_cos(self, px_size, method, limit_z_fit = True, debug = False):
        """
        Find direction cosines for the beam:
        e_x = v_x / |v|
        e_y = v_y / |v|
        e_z = v_z / |v|

        Find tilt of the beam with respect to z axis.

        :params method: "stats" or "fit" - specify how the coordinates of the beam (mu) and respective spread (sigma)
        are calculated.
        :params debug: if debug True, the linear fits to the beam coordinates are plotted
        """

        if limit_z_fit:  # excludes trajectory points beyond a certain z value.
            try:
                max_z_idx = self.max_z_idx_fit
            except AttributeError:
                print(f"Please define the max_z_id with tomo.set_max_z() to limit the fitting to valid cross sections.")
        else:
            max_z_idx = len(self.beam_coord_l)

        # organise data
        try:
            x_arr = np.array(self.beam_coord_l)[:max_z_idx, 0]
            y_arr = np.array(self.beam_coord_l)[:max_z_idx, 1]
            z_arr = np.array(self.beam_coord_l)[:max_z_idx, 2] * (10 ** -3) / px_size  # z_arr in same units [px]
            t_arr = np.arange(0, len(x_arr))

        except (ValueError, TypeError, IndexError) as e:
            print(30 * "#")
            print(f"No beam was assigned to the provided grid point: id_x = {self.id_x}, id_y = {self.id_y}.")
            print("Generated error: ", e)
            return None

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
            ax[0].set_title(f"v_x = {v_x:.2f}, e_x = {e_x:.2f}\nmethod={method}")
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

    def find_div(self, pixel_size, method, limit_z_fit = True, debug = False):
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

        try:
            z_arr = beam_coord[:max_z_id, 2]
            z_px_arr = (z_arr * 10 ** -3) / pixel_size
            beam_width_l = np.array(self.beam_width_l)
            width_col_arr = beam_width_l[:max_z_id, 0] #col_sigma,
            width_row_arr = beam_width_l[:max_z_id, 1] #row_sigma

        except (TypeError, ValueError, IndexError):
            print(30 * "#")
            print(f"No beam assigned to grid point id_x = {self.id_x} and id_y = {self.id_y}.")
            print("Generated error: ", e, "in Beam.find_div()")
            return None

        # y - fit
        linear_f = lambda x, m, b: m * x + b
        popt, _ = curve_fit(linear_f, z_px_arr, width_col_arr, )
        width_col_fit_arr = linear_f(z_px_arr, *popt)
        m_y = popt[0]
        theta_col = np.degrees(np.arctan(m_y))

        # x - fit - sum over col
        popt, _ = curve_fit(linear_f, z_px_arr, width_row_arr, )
        width_row_fit_arr = linear_f(z_px_arr, *popt)
        m_x = popt[0]
        theta_row = np.degrees(np.arctan(m_x))

        # Update beam_i.attributes
        self.div_row = 2*theta_row #full angle = 2 * half-angle
        self.div_col = 2*theta_col

        # Plot
        if debug:
            print(f"The divergence angles of the beam idx={self.id_x}, idy = {self.id_y} have been updated:")
            print(f"div_row = {self.div_row:.3f}deg, e_col = {self.div_col:.3f}deg")

            f, ax = plt.subplots()
            plt.title(
                f"Debug - find_div - method = {method},\n m_x = {m_x:.2f}, 2 x theta_row = {2*theta_row:.2f}deg ,\nm_y = {m_y:.2f}, 2 x theta_col = {2*theta_col:.2f}deg,")
            # ax.plot(z_px_arr, width_x_arr, label = "x")
            ax.plot(z_px_arr, width_col_arr, ".", label="y", color = "tab:blue")
            ax.plot(z_px_arr, width_col_fit_arr, label="y-fit", color = "tab:blue")

            ax.plot(z_px_arr, width_row_arr, ".", label="x", color = "tab:orange")
            ax.plot(z_px_arr, width_row_fit_arr, label="x-fit", color = "tab:orange")
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

    def get_centroid_val(self, signal):
        """
        Calculate the centroid of a signal --> y_arr is proportional to the occurrence probability
        """
        idx_arr = np.arange(len(signal))
        signal = np.abs(signal)
        prob_arr = signal / np.sum(signal)  # probability mass function
        centroid = np.sum(prob_arr * idx_arr)
        return centroid

    def get_sigma_val(self, signal):
        """
        Calculate the centroid of a signal --> y_arr is proportional to the occurrence probability
        """
        idx_arr = np.arange(len(signal))
        signal = np.abs(signal)
        prob_arr = signal / np.sum(signal)  # probability mass function
        centroid = np.sum(prob_arr * idx_arr)

        sigma = np.sqrt(np.sum(prob_arr * ((idx_arr - centroid) ** 2)))

        return sigma

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

    def get_roi_specs(self, roi_img, roi_width, method = "fit", sub_backg = False, debug=False):
        """
        Given a 2D array with a region of interest (ROI) around a beam,
        fit a gaussian beam to the marginal X and marginal Y distributions
        and retrieve the middle point for X and Y and the sigma of the
        gaussian fit.


        In the case fit=False or if the gaussian fit fails, we extract the
        statistics of the intensity distribution: centroid_x, sigma_x, centroid_y, sigma_y

        :param roi_img: 2D array with region of interest around beam
        :type roi_img: numpy array (2D)

        :param roi_width: width of region of interest in pixel units;
        :type roi_width: int

        :param fit: if True, extract params of gaussian fit. If false, extract params
        from statistical distribution of intensity.
        :type fit: boolean

        :return: i_row, i_col, row_mu, col_mu, row_sigma, col_sigma
        :
        """

        # Project intensity map of ROI into x and y --> I(x) = roi_sum_col, I(y) = roi_sum_row
        roi_sum_row_arr = np.sum(roi_img, axis=0)
        roi_sum_col_arr = np.sum(roi_img, axis=1)

        #if debug:
        #    plt.figure()
        #    plt.title("Debug beam_i.get_roi_specs- ROI of beam")
        #   plt.imshow(roi_img, origin = "lower")

        #get x arr for plotting
        row_idx_arr = range(len(roi_sum_row_arr))
        col_idx_arr = range(len(roi_sum_col_arr))

        # Normalise I(x) and I(y)
        roi_sum_row_arr = normalise_arr(roi_sum_row_arr)
        roi_sum_col_arr = normalise_arr(roi_sum_col_arr)

        # Correct background - background has a slope --> mx+b --> subtract background(x) from I(x)
        if sub_backg: #subtract background level from roi?
            roi_sum_row_arr = self.subtract_lin_backg(roi_sum_row_arr, debug=debug)
        roi_sum_row_arr = normalise_arr(roi_sum_row_arr)

        if sub_backg: #subtract background level from roi?
            roi_sum_col_arr = self.subtract_lin_backg(roi_sum_col_arr, debug=debug)
        roi_sum_col_arr = normalise_arr(roi_sum_col_arr)

        i_row = roi_sum_row_arr #sum along cols
        i_col = roi_sum_col_arr


        if method == "fit":
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

            except (RuntimeError, ValueError, TypeError):
                # if the fit fails, extract statistics
                print(f"Gaussian fit failed for beam (id_x, id_y) = ({self.id_x},{self.id_y}). Beam position and width",
                      f"were retrieved by stastical method")
                row_mu = self.get_centroid_val(i_row)
                row_sigma = self.get_sigma_val(i_row)
                col_mu = self.get_centroid_val(i_col)
                col_sigma = self.get_sigma_val(i_col)
                popt_row = [row_mu, row_sigma]
                popt_col = [col_mu, col_sigma]

        elif method == "stats": # --> extract statistical mu and sigma directly. No fit is used
            row_mu = self.get_centroid_val(i_row)
            row_sigma = self.get_sigma_val(i_row)
            col_mu = self.get_centroid_val(i_col)
            col_sigma = self.get_sigma_val(i_col)
            popt_row = [row_mu, row_sigma]
            popt_col = [col_mu, col_sigma]

        else:
            print("Please choose a tomography method: 1. fit or 2. stats")

        if debug:
            plt.figure()
            plt.title("Debug - beam_i.get_roi_specs()")
            plt.plot(row_idx_arr, roi_sum_row_arr, ".", color="k")
            y_fit = gaussian(row_idx_arr, row_mu, row_sigma)
            plt.plot(row_idx_arr, y_fit, label="row")
            plt.xlabel("[px]")

            plt.plot(col_idx_arr, roi_sum_col_arr, ".", color="k")
            y_fit = gaussian(row_idx_arr, col_mu, col_sigma)
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
            try:
                max_z_id = self.max_z_idx_fit
            except AttributeError:
                print("The beam.max_z_idx_fit attribute has not been set. ","\n",
                      "The trajectory will be plotted with all cross sections.","\n",
                      "If you wish to define beam.max_z_idx_fit, please run tomo.set_max_z(val)")
                max_z_id = len(self.beam_coord_l)
        else:
            max_z_id = len(self.beam_coord_l)

        try:
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
        except (TypeError, ValueError, IndexError) as e:
            print(f"Caught exception {e}.")
            print(f"No beam was associated with the provided grid point.")

    def plot_width(self, limit_z_fit = True):
        beam_coord = np.array(self.beam_coord_l)

        if limit_z_fit: #excludes trajectory points beyond a certain z value.
            try:
                max_z_id = self.max_z_idx_fit
            except AttributeError:
                print(f"Please define the max_z_id with tomo.set_max_z() to limit the fitting to valid cross sections.")
        else:
            max_z_id = len(self.beam_coord_l)

        try:
            z_arr = beam_coord[:max_z_id, 2]
            beam_width_l = np.array(self.beam_width_l)
            width_x_arr = beam_width_l[:max_z_id, 0]
            width_y_arr = beam_width_l[:max_z_id, 1]
        except IndexError as e:
            print(30*"#")
            print(f"No beam was assigned to the provided grid point: id_x = {self.id_x}, id_y = {self.id_y}.")
            print("Generated error: ", e)
            return None

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
            try:
                z_i = self.beam_coord_l[i][2]
            except TypeError as e:
                print(30 * "#")
                print(f"No beam was assigned to the provided grid point: id_x = {self.id_x}, id_y = {self.id_y}.")
                print("Generated error: ", e)
                return None

            f, ax = plt.subplots()
            plt.imshow(roi_i)
            plt.colorbar()
            plt.title(f"z_idx = {i}, z = {z_i:.2f}mm")
            plt.xlabel("id_y [px]")
            plt.ylabel("id_x [px]")

            return f, ax

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
            try:
                roi_i = self.roi_l[id_z]
                col_mu, row_mu, col_sigma, row_sigma = self.roi_beam_param_l[id_z]  # fit params
                i_col_arr = self.i_col_l[id_z]  # sum over columns
                i_row_arr = self.i_row_l[id_z]  # sum over lines
                idx_arr = np.arange(len(i_col_arr))
            except (ValueError, TypeError) as e:
                print(30 * "#")
                print(f"No beam was assigned to the provided grid point: id_x = {self.id_x}, id_y = {self.id_y}.")
                print("Generated error: ", e)
                return None

            # Plots
            f, ax = plt.subplots(ncols=2, nrows=1, figsize=(8, 4))
            ax[0].imshow(roi_i, origin="lower")
            ax[0].set_xlabel("Y [px]")
            ax[0].set_ylabel("X [px]")
            ax[0].set_title(f"id_x = {self.id_x},  id_y = {self.id_y}, id_z = {id_z}")

            # plot sum_col - x
            ax[1].plot(idx_arr, i_col_arr, ".", label="sum_col - y", color="tab:blue")
            col_fit = gaussian(idx_arr, col_mu, col_sigma)
            ax[1].plot(idx_arr, col_fit, color="tab:blue")

            # plot sum_col - y
            ax[1].plot(idx_arr, i_row_arr, ".", label="sum_row - x", color="tab:orange")
            row_fit = gaussian(idx_arr, row_mu, row_sigma)
            ax[1].plot(idx_arr, row_fit, color="tab:orange")

            ax[1].set_title(
                f"$\mu_x = {{{col_mu:.1f}}},  \sigma_x = {{{col_sigma:.1f}}}$ \n $\mu_y = {{{row_mu:.1f}}},  \sigma_y = {{{row_sigma:.1f}}}$")
            ax[1].set_xlabel("Pos [px]")
            ax[1].legend()

    def plot_beam_params(self, limit_z_fit = False):
        """
        Plot for each cross section the intensity marginal distributions, showing the position of the intensity
        centroid (mu) and the intensity spread (sigma). This is particularly meaningful, when the beam parameters
        are determined using the statistical method, instead of a gaussian fit.

        :param limit_z_fit: only plots cross section with z lower than max_id_z
        """

        if limit_z_fit: #excludes trajectory points beyond a certain z value.
            try:
                max_id_z = self.max_z_idx_fit
            except AttributeError:
                print(f"Please define the max_z_id with tomo.set_max_z() to limit the fitting to valid cross sections.")
        else:
            max_id_z = len(self.beam_coord_l)

        for id_z in range(max_id_z):
            try:
                roi_i = self.roi_l[id_z]
                col_mu, row_mu, col_sigma, row_sigma = self.roi_beam_param_l[id_z]  # fit params
                i_col_arr = self.i_col_l[id_z]  # sum over columns
                i_row_arr = self.i_row_l[id_z]  # sum over lines
                idx_arr = np.arange(len(i_col_arr))
            except (ValueError, TypeError) as e:
                print(30 * "#")
                print(f"No beam was assigned to the provided grid point: id_x = {self.id_x}, id_y = {self.id_y}.")
                print("Generated error: ", e)
                return None

            # Plots
            f, ax = plt.subplots(ncols=2, nrows=1, figsize=(8, 4))
            ax[0].imshow(roi_i, origin="lower")
            ax[0].set_xlabel("Y [px]")
            ax[0].set_ylabel("X [px]")
            ax[0].set_title(f"id_x = {self.id_x},  id_y = {self.id_y}, id_z = {id_z}")

            # plot sum_row - x
            ax[1].plot(idx_arr, i_col_arr, ".", label="sum_col - y", color="tab:blue")
            ax[1].axvline(x=col_mu, color="tab:blue", linestyle="-", linewidth = 3)
            ax[1].axvline(x=col_mu + col_sigma, color="tab:blue", linestyle="--")
            ax[1].axvline(x=col_mu - col_sigma, color="tab:blue", linestyle="--")

            # plot sum_col - y
            ax[1].plot(idx_arr, i_row_arr, ".", label="sum_row - x", color="tab:orange")
            ax[1].axvline(x=row_mu, color="tab:orange", linestyle="-", linewidth = 3)
            ax[1].axvline(x=row_mu - row_sigma, color="tab:orange", linestyle="--")
            ax[1].axvline(x=row_mu + row_sigma, color="tab:orange", linestyle="--")

            ax[1].set_title(
                f"$\mu_x = {{{col_mu:.1f}}},  \sigma_x = {{{col_sigma:.1f}}}$ \n $\mu_y = {{{row_mu:.1f}}},  \sigma_y = {{{row_sigma:.1f}}}$")
            ax[1].set_xlabel("Pos [px]")
            ax[1].legend()

    def find_pos_and_widths(self, cross_sect_l, cross_sect_z_l, roi_width, method = "fit", sub_backg = False, debug = False):

        """
        Extracts the coordinates and width of the beam, by analysing its trajectory across
        several cross-sections.

        params:
        :param cross_sect_l: list with CrossSection objects;
        :type cross_sect_l: list

        :param cross_sect_z_l: list containing the z-coordinates of each cross section;
        :type cross_sect_z_l: list

        :param roi_width: width in pixels of the region of interest (square)
        :type roi_width: integer

        debug: plots ROI of each beam

        :param fit: True if the beam positions and widths are extracted with a gaussian fit method
        and False is a statistical method is used.
        :type fit: bool

        """
        n_sections = len(self.beam_coord_l)

        for id_z in range(n_sections):
            #define initial_guess for center of ROI (region of interest)
            try:
                if id_z == 0:
                    pos_idx_x, pos_idx_y, pos_id_z = self.beam_coord_l[id_z]
                else:
                    pos_idx_x, pos_idx_y, pos_id_z = self.beam_coord_l[id_z-1] #use beam position from previous cross sect

            except (ValueError, TypeError): # if point of the grid, does not have a beam
                print(f"No beam assigned to grid point id_x = {self.id_x} and id_y = {self.id_y}.")
                return None

            #get beam roi, based on coordinates of beam in lower section
            image_i = cross_sect_l[id_z].image_rot
            roi_i = get_roi(image_i, pos_idx_x, pos_idx_y,
                            roi_width = roi_width)

            roi_i = gaussian_filter(roi_i, sigma = int(roi_width / 20), )  # smoothen ROI for pixel noise

            # establish new coordinates of beam in current z_id, with gaussian fit
            #row_mu corresponds to a sum over rows --> I(y)
            #col_mu corrsponds to a sum over cols --> I(x)
            i_row, i_col, row_mu, col_mu, row_sigma, col_sigma = self.get_roi_specs(roi_i, roi_width, method = method, sub_backg = sub_backg, debug=debug)
            pos_idx_x = int(pos_idx_x + (col_mu - roi_width / 2)) #col_mu --> summed over colums --> x
            pos_idx_y = int(pos_idx_y + (row_mu - roi_width / 2))
            pos_z = cross_sect_z_l[id_z]

            if debug:
                plt.figure()
                plt.title(f"Debug - beam_i.complete_coords_widths,\nz= {pos_z:.2f}mm")
                plt.imshow(roi_i, origin="lower")

            # update beam.roi_l and beam.beam_coord_l
            self.beam_coord_l[id_z] = np.array([pos_idx_x, pos_idx_y, pos_z]) # units are [px, px, mm]
            self.beam_width_l[id_z] = np.array([col_sigma, row_sigma]) #row_sigma --> sum over rows --> I(y)
            self.beam_intensity_l[id_z] = np.max(roi_i)

            self.roi_l[id_z] = roi_i
            self.roi_beam_param_l[id_z] = [col_mu, row_mu, col_sigma, row_sigma]
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
tomo.find_pos_and_widths(debug = False) #Find coords for all the other layers

#Extract parameters, direction cosines and divergence for each beam
tomo.set_max_z(16.0) #limits fits to cross section with z<max_z;
tomo.find_dir_cos_div(limit_z_fit = True, debug = False)

#Visualise data
tomo.plot_div() #plots array with divergences in x and y
tomo.plot_dir() #plots the direction of each beam in an array of polar plots
tomo.plot_dir_single() #single plot for all beams

"""
