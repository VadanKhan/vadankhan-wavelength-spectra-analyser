@echo off
cd /d "C:\Users\762093\Documents\vadankhan-wavelength-spectra-analyser"
start /min cmd /c ".venv\Scripts\activate.bat && python vadankhan_wavelength_spectra_analyser\spectrum_files_transform_watchdog.py"