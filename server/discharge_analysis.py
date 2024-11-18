import io
import base64
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize

def read_raster_data(file_path):
    """Reads raster data from a TIFF file."""
    with rasterio.open(file_path) as src:
        data = src.read(1)  # Read the first band
        meta = src.meta
    return data, meta

# Function to encode plot to base64
def encode_plot_to_base64():
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    return f"data:image/png;base64,{base64.b64encode(image_png).decode()}"

# Function to display a map and return the base64 string
def display_map_base64(data, title, color_label):
    plt.figure(figsize=(10, 6))
    plt.imshow(data, cmap='viridis')
    plt.colorbar(label=color_label)
    plt.title(title)
    plt.axis('equal')
    return encode_plot_to_base64()

def get_cell_area():
    """Calculates the area of each cell in the raster (in square meters)."""
    with rasterio.open('clipped_ndvi_crs.tif') as src:
        transform = src.transform
        pixel_width = transform[0]   # Width of a pixel in x-direction
        pixel_height = -transform[4]  # Height of a pixel in y-direction
        pixel_area = pixel_width * pixel_height  # Area in square meters
    return pixel_area

def calculate_discharge(runoff_data, rainfall_data, cell_area):
    """Calculates the discharge data using the formula Q = C * I * A."""
    # Convert total annual rainfall from millimeters to inches
    rainfall_data_in_inches = rainfall_data * 0.0393701  # 1 mm = 0.0393701 inches
    # Convert inches per year to inches per hour
    rainfall_data_in_inches_per_hour = rainfall_data_in_inches / 8760  # 8760 hours in a year
    # Convert cell area to acres (1 acre = 4046.86 m^2)
    cell_area_in_acres = cell_area / 4046.86
    # Calculate the discharge Q = C * I * A
    discharge_data = runoff_data * rainfall_data_in_inches_per_hour * cell_area_in_acres
    # Ensure no negative values
    discharge_data[discharge_data < 0] = 0
    return discharge_data

def perform_discharge_analysis(runoff_tif_path, rainfall_tif_path):

        
    # Read the runoff coefficient and rainfall intensity .tif files
    runoff_data, runoff_meta = read_raster_data(runoff_tif_path)
    rainfall_data, rainfall_meta = read_raster_data(rainfall_tif_path)


    # Convert data to double precision
    runoff_data = runoff_data.astype(np.float64)
    rainfall_data = rainfall_data.astype(np.float64)

    # Resample rainfall data to match runoff data if necessary
    if runoff_data.shape != rainfall_data.shape:
        rainfall_data = resize(rainfall_data, runoff_data.shape, mode='reflect', anti_aliasing=True)

    # Get cell area
    cell_area = get_cell_area()
    print(f'Cell Area: {cell_area:.2f} square meters')

    # Calculate discharge
    discharge_data = calculate_discharge(runoff_data, rainfall_data, cell_area)

    # Calculate total discharge (units: inches * acres / hour)
    total_discharge = np.nansum(discharge_data)

    # Convert total discharge to cubic feet per second
    # 1 inch * acre = 3630 cubic feet
    total_discharge_cubic_feet_per_hour = total_discharge * 3630
    total_discharge_cfs = total_discharge_cubic_feet_per_hour / 3600  # Convert to per second

    print(f'Total Discharge: {total_discharge_cfs:.2f} cubic feet per second')
    # Generate plots
    plots = {
        'runoff_coeff': display_map_base64(runoff_data, 'Runoff Data', 'Runoff Coefficient (C)'),
        'rainfall': display_map_base64(rainfall_data, 'Rainfall Data (Resampled)', 'Rainfall (mm/year)'),
        'discharge_map': display_map_base64(discharge_data, 'Discharge Map (Q = C * I * A)', 'Discharge (in*acre/hr)')
    }

    return {
        'plots': plots,
        'total_discharge': total_discharge_cfs,
        'reservoir_area': cell_area
    }
