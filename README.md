# VL Chip emission analysis

This tool is designed to analyze the properties of a grid of laser beams emitted from Vitrealab light chips. It focuses on evaluating:
1. The intensity distribution of the grids of beams; 
2. The beams divergence in degrees;
3. The propagation direction of each beam within the array. 

The code structure is primarily centered around three main classes: `Tomography`, `CrossSection`, and `Beam`. Each class is tailored to handle different aspects of the beam analysis process, from loading and processing data to calculating specific beam properties.

## 1. Tomography Class

### Purpose
The `Tomography` class serves as the backbone of the analysis tool. It manages the overall process, including loading data, determining beam positions and widths across multiple cross-sections, and calculating the beams' direction cosines and divergence angles.

### Attributes
- **filename, shape, roi_width, sub_backg**: Basic setup information, including the data file, grid shape, region of interest (ROI) width, and background subtraction preference.
- **cross_sect_image_l, cross_sect_z_l, cross_sect_l**: Lists storing images, z-coordinates of cross-sections, and `CrossSection` objects, respectively.
- **beam_l**: A grid (list of lists) storing `Beam` objects, representing each beam in the array.
- **pixel_size, max_z_fit, mean_dir_cos**: Various parameters for analysis, including the pixel size of images, maximum z-coordinate for fitting procedures, and the mean direction cosine of all beams.

### Methods
- **load_data, find_rot_spacing, init_coords**: Functions for loading the data, finding the optimal rotation angle for images, and initializing the coordinates of beams in the first layer.
- **find_pos_and_widths, set_max_z, find_dir_cos, find_div, find_mean_dir_cos**: Methods for calculating positions, widths, direction cosines, and divergence of beams.
- **plot_**: Various plotting functions to visualize results, such as direction and divergence of beams.

## 2. CrossSection Class

### Purpose
The `CrossSection` class encapsulates information and operations related to individual cross-sections of the beam array, such as image processing and peak detection.

### Attributes
- **z_coord, shape, image**: Information about the cross-section's z-coordinate, shape (number of rows and columns), and the image data.
- **spacing_px, rot_angle, image_rot**: The pixel spacing, rotation angle to align beams vertically, and the rotated image.
- **beam_coord_l**: Coordinates of beam centroids within the cross-section.

### Methods
- **simple_plot, find_rot, find_peaks**: Functions for plotting the cross-section, finding the optimal rotation angle, and detecting beam peaks.
- **find_geom, id_to_coord**: Methods for calculating geometric properties and converting beam indices to coordinates.

## 3. Beam Class

### Purpose
The `Beam` class focuses on analyzing individual beams, including their trajectory, width, intensity, and direction across multiple cross-sections.

### Attributes
- **id_x, id_y**: Identifiers for the beam's position in the grid.
- **beam_coord_l, beam_width_l, beam_intensity_l**: Lists storing the coordinates, widths, and maximum intensity of the beam at different z-coordinates.
- **roi_l, e_x, e_y, e_z, div_row, div_col**: ROI images of the beam, direction cosines, and divergence angles in row and column directions.

### Methods
- **find_dir_cos, find_div**: Calculate the direction cosines and divergence of the beam.
- **get_centroid_val, get_sigma_val, find_lin_backg, subtract_lin_backg, get_roi_specs**: Utility functions for analyzing the beam's ROI and extracting statistical properties.
- **plot_trajectory, plot_width, plot_rois, plot_gauss_fit, plot_beam_params**: Visualization methods for various beam properties.

## Overall Workflow

1. **Data Loading**: The `Tomography` object is initialized with the dataset, loading images and their z-coordinates.
2. **Beam Analysis**: Beam positions and properties are calculated for each cross-section using methods in the `Tomography` class, which internally uses `CrossSection` and `Beam` objects for detailed analysis.
3. **Result Visualization**: Various plotting methods are provided to visualize the direction and divergence of beams, as well as other properties of interest.

This tool offers a comprehensive approach to analyzing laser beams, with a focus on flexibility and detail in evaluating beam characteristics.
