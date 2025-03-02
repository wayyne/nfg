# nfg

roughness map generator

For context please see:  
*N. Rogic, S.J. Charbonnier, F. Garin, G.W. Dayhoff II, E. Gagliano, M. Rodgers, C.B. Connor, S. Varma, D. Shean, 2023. "Characterizing and Mapping Volcanic Flow Deposits on Mount St. Helens via Dual-band SAR Imagery", submitted to Remote Sensing, March 2023*


This software has non-standard dependencies, we recommend setting up a python virtual enviroment that satisfies the dependencies. Specificially, numpy and gdal with osgeo are required. Numpy installation is straight-forward; for osgeo installation see: https://gdal.org/api/python_bindings.html

Software is provided as is, a manuscript detailing the implementation, use-cases, and modularity, along with a user-manual, is in preperation.

# Roughness Map Generator (NFG) README

## Overview
This repository contains the Roughness Map Generator (NFG), a Python script designed to process geospatial raster data (.tif files) and compute roughness maps. The tool handles tasks such as converting raw amplitude data to backscatter values, applying various roughness models over sliding windows, and performing optional post‐processing like rescaling, clustering, and intensity weighting. Multiprocessing is supported to improve performance on large datasets.

## Requirements
- Python 3.x
- NumPy
- GDAL (with Python bindings)
- OSGeo (for spatial reference operations)

## Installation
1. Ensure that Python 3 and pip are installed.
2. Install required packages:
    ```bash
    pip install numpy pip install gdal
    ```
   
(Other system dependencies may be necessary depending on your platform.)

## Usage
Run the script from the command line:
  ```bash
  ./nfg.py [INPUT_PATH] [WINDOWS] [ROUGHNESS_MODEL]
  ```

- **INPUT_PATH**: Path to a single .tif file or a directory containing .tif files.
- **WINDOWS**: Comma-separated list of window sizes (e.g., "3,5,7").
- **ROUGHNESS_MODEL**: Integer corresponding to the roughness model to use (see details below).

If no command-line arguments are provided, the script will use the default parameters defined within the file.

## Run Parameters

### Input/Output Parameters

#### INPUT_PATH
Specifies the data source. It can be a single .tif file containing backscatter amplitude data or a directory with multiple .tif files. When a directory is specified and AVERAGE_INPUTS is true, the outputs from each file are averaged.

#### ANGLE_FILE
(Optional) Path to a text file containing latitude, longitude, slant angle, and incidence angle values. If provided, these angles are used during the conversion of raw amplitude data to backscatter values.

### Data Conversion Parameters

#### PINCH_INPUT
A Boolean flag that enables clipping (or “pinching”) of the input data to constrain its range based on a specified pinch factor.

#### CONVERT_INPUT_TO_BS
Toggle that, when true, converts the amplitude values from the input into backscatter values using logarithmic equations.

#### USE_OG_BS_EQN
Flag to select the conversion equation. When true, the original NASA backscatter conversion equation is used; otherwise, an alternative equation incorporating incidence angles is applied (if ANGLE_FILE is available).

#### IMAGE_NOISE_BIAS
A numeric parameter representing the noise bias present in the raw data. This value is subtracted from the squared amplitude before the logarithmic conversion to backscatter.

#### SPLIT_OUTPUT
Boolean flag that, when enabled, produces two separate output files. The split is based on a threshold value, dividing the data into two classes.

#### SPLIT_THRESHOLD
Numeric threshold used when SPLIT_OUTPUT is enabled. Pixels are separated into two outputs based on whether they are above or below this threshold.

### Output Processing Parameters

#### COHERENT_OUTPUT
When enabled, a directory of inputs is treated as a single coherent input for purposes such as rescaling. This affects how the overall output is computed from multiple files.

#### RESCALE_OUTPUT
A Boolean flag that rescales the final output to the range [0, 1]. This is useful for normalizing the roughness values.

#### CLUSTER_OUTPUT
Toggle that performs binary clustering of the final output based on a defined class boundary, effectively classifying regions as “smooth” or “rough.”

#### CLASS_BOUNDARY
A numeric value that sets the boundary between smooth and rough classes in the clustering process.

#### DUMP_DATA
If true, a .dat file is generated that contains a column of all final pixel values from the output.

#### DUMP_NODATA
When enabled, pixels marked with the NODATA_VALUE are included in the dumped data file.

### Roughness Model Parameters

#### ROUGHNESS_MODEL
Specifies the index (from 0 to 9) of the roughness model to apply. Each model (m0 to m9) implements a different statistical measure (e.g., standard deviation, mean error, RMSE, etc.) calculated over a window.

#### WINDOWS
A list containing one or more window sizes (NxN) used for computing roughness. Multiple window sizes allow for multi-pass analysis, with the final output typically averaged over all specified windows.

#### APPLY_GAUSSIAN_BLUR
Boolean flag that, when true, applies a 2D Gaussian blur to the input data before computing roughness. This can help smooth the data.

#### BLUR_FACTOR
A numeric value used in the Gaussian kernel. It is typically set to 1 divided by the square root of 2π, ensuring normalization of the filter response for the central pixel.

#### REPORT_AS_LOG10
When enabled, the computed roughness values are reported as their base-10 logarithms. This is useful when the dynamic range of the data is large.

#### POPULATION_STATISTICS
Toggle that determines whether to use population statistics (dividing by N²) or sample statistics (dividing by N²–1) in standard deviation and variance calculations.

### Intensity Weighting Parameters

#### INTENSITY_WEIGHTED
Flag that enables intensity weighting of the final output. In this process, inverted backscatter values are used to weight the roughness measure.

#### PINCH_WEIGHTS
Boolean flag that applies the pinch factor to the intensity weights, constraining them within a specific range.

#### PINCH_FACTOR
A numeric parameter representing the number of standard deviations used to clip (pinch) both the input data and the intensity weights. It limits extreme values to improve the stability of the conversion and weighting processes.

### Missing Data Handling Parameters

#### NODATA_VALUE
Specifies the value used to indicate missing or invalid data in the input and output datasets. Computations will ignore pixels with this value.

#### NODATA_ANGLE
A special numeric value used in the angle file to denote missing or invalid angle data.

#### AVOID_BLANKS
When true, the script avoids computing roughness in any window that contains missing (blank) data, ensuring that results are not skewed by invalid values.

#### WORKAROUND_BLANKS
If enabled, the script will attempt to reduce the window size recursively to work around missing data rather than simply skipping the computation.

### Performance Parameters

#### MULTIPROCESSING
A Boolean toggle that enables multiprocessing. When true, the computation of roughness over the raster is distributed across available CPU cores, significantly reducing processing time for large datasets.

## Script Structure and Workflow

1. **Data Input and Preparation:**  
   - Reads input .tif files (or a directory of files) and optionally an angle file.
   - Validates data and handles missing values using the NODATA_VALUE.
   - Optionally converts raw amplitude data to backscatter using the specified conversion parameters.

2. **Roughness Calculation:**  
   - Applies the selected roughness model (m0 to m9) over a sliding window across the dataset.
   - Uses multiprocessing if enabled, splitting the work among CPU cores.
   - Provides options to compute statistics in log scale or with population/sample statistics.

3. **Post-Processing:**  
   - Applies optional intensity weighting and data “pinching.”
   - Can rescale the output to a normalized range and perform binary clustering based on a class boundary.
   - Splits the output into separate files if SPLIT_OUTPUT is enabled.

4. **Output Generation:**  
   - Writes the processed roughness map to a .tif file, preserving geospatial metadata.
   - Generates additional outputs (e.g., log files, .dat dumps) as configured.

## Logging and Progress Reporting
- The script writes progress messages both to the console and to a temporary log file.
- It reports estimated run times, statistical summaries of the input and output, and other relevant processing milestones.

## Acknowledgments
Developed by Nikola Rogic, Ph.D., Franco Villegas-Garin, and Guy Dayhoff II  
Version 38 (December 2021 – May 2022)

## License
[Insert License Information Here]

## Contact
For additional information or support, please contact [Your Contact Information].
