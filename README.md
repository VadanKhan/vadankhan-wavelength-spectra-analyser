# Wavelength Spectra Processing Script

## Overview
This script is designed to efficiently process large spectral data files by extracting and transforming wavelength intensity data for a given wafer. It converts the raw data from a wide format to a long format while filtering the relevant wavelength range, allowing for optimized storage and analysis.

## Optimization Techniques
- **Chunked Processing**: The script reads and processes large CSV files in chunks (default: 1000 rows at a time) to reduce memory usage.
- **Column Filtering**: Only relevant intensity columns within a specified wavelength range are loaded, minimizing unnecessary data processing.
- **Generator Based Streaming to CSV**: Instead of storing all processed data in memory, the script writes directly to disk in chunks to prevent memory overload.
- **Parallel Execution (Experimental)**: A threading-based parallel execution approach is being tested to further speed up processing.

## How to Use
First note that **Raw spectra files from LIV need to be manually downloaded** into the `\wavelength_spectra_files` directory. **AND** the files need to be renamed to have the wafer code in the name somewhere, so that the code can find the appropriate files. 

To run the script efficiently, execute only the **first three cells** in the Jupyter Notebook: `vadankhan_wavelength_spectra_analyser\spectrum_files_transform.ipynb`:
1. **Imports and Paths**: Sets up the required directories and imports the required packages.
2. **Define Data Import Locations**: Searches the wavelength_spectra_files folder for appropriate input files. 
3. **Run the Processing Pipeline**: Calls the main processing function to handle the data transformation and export (to the `exports` folder )

## Experimental Multi-Threading (Optional)
The fourth cell in the notebook contains an experimental **threading-based parallel execution** approach. It attempts to process multiple files simultaneously to improve performance. However, since it is still being tested, results may vary, and unexpected behavior could occur. If you experience issues, use the standard sequential processing method.

