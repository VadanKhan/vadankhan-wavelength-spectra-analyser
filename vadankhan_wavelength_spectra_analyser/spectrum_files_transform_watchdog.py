import os
import sys
import warnings
from pathlib import Path
import time
import csv
import io
import re
from datetime import datetime
import threading

# import threading
# import multiprocessing
# import concurrent.futures

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dask.dataframe as dd
import polars as pl
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# import pyarrow as pa
# import pyarrow.csv as csv
from scipy.signal import find_peaks

# import requests
# from bs4 import BeautifulSoup

CURRENT_DIR = Path(os.getcwd())
# Move to the root directory
ROOT_DIR = CURRENT_DIR

RAW_FILE_PATH = ROOT_DIR / "wavelength_spectra_files"
RAW_FILE_PATH.mkdir(parents=True, exist_ok=True)
EXPORTS_FILE_PATH = ROOT_DIR / "exports"
EXPORTS_FILE_PATH.mkdir(parents=True, exist_ok=True)

# Add the root directory to the system path

sys.path.append(str(ROOT_DIR))


ANALYSIS_RUN_NAME = "debug-watchdog-script"

SUBARU_DECODER = "QC WAFER_LAYOUT 24Dec.csv"
HALO_DECODER = "HALO_DECODER_NE-rev1_1 logic_coords_annotated.csv"


# ------------------------------------ Spectrum Analysis Code ------------------------------------ #
def load_decoder(decoder_file_path):
    print(f"Loading decoder from: {decoder_file_path}")
    start_time = time.time()

    if not decoder_file_path.exists():
        print(f"Decoder file not found at {decoder_file_path}")
        return pd.DataFrame()

    df_decoder = pd.read_csv(decoder_file_path, usecols=["Logic_X", "Logic_Y", "TE_LABEL", "TYPE"])
    df_decoder = df_decoder.set_index(["Logic_X", "Logic_Y"])

    end_time = time.time()
    print(f"Loaded in {end_time - start_time:.2f} seconds.\n")
    return df_decoder


def transform_raw_file(
    filepath, wafer_id, decoder_df, wavelength_lb=824, wavelength_ub=832, chunksize=1000, max_chunks=10
):
    print(f"Starting file transformation for {wafer_id}...")
    total_t0 = time.time()

    t1 = time.time()
    col_names = pd.read_csv(filepath, nrows=1).columns
    intensity_cols = [col for col in col_names if col.startswith("Intensity_")]
    wavelengths = {col: float(col.split("_")[1]) for col in intensity_cols}
    selected_intensity_cols = [col for col, wl in wavelengths.items() if wavelength_lb <= wl <= wavelength_ub]
    usecols = ["X", "Y"] + selected_intensity_cols
    data_points_threshold = len(selected_intensity_cols)
    print(f"Header parsing and column filtering took {time.time() - t1:.2f} s")

    with pd.read_csv(filepath, chunksize=chunksize, usecols=usecols) as reader:
        for i, chunk in enumerate(reader):
            if i >= max_chunks:
                break

            chunk_start = time.time()

            # Base transformation
            t_base = time.time()
            long_df = chunk.melt(
                id_vars=["X", "Y"], value_vars=selected_intensity_cols, var_name="Wavelength", value_name="Intensity"
            )
            long_df["Wavelength"] = long_df["Wavelength"].map(wavelengths)
            long_df = long_df.merge(decoder_df, left_on=["X", "Y"], right_index=True, how="left")
            long_df = long_df.drop(columns=["X", "Y"])
            long_df = long_df[["TYPE", "TE_LABEL", "Wavelength", "Intensity"]]
            t_base_elapsed = time.time() - t_base

            yield long_df, data_points_threshold, t_base_elapsed

            # print(f"Chunk {i+1} | Base transform: {t_base_elapsed:.2f}s | Total time: {time.time() - chunk_start:.2f}s")

    print(f"File transformation for {wafer_id} completed in {time.time() - total_t0:.2f} seconds.")


def extract_top_two_peaks(df_group):
    """
    Detects the top two peaks in a spectrum.
    Returns:
        peak_series (pd.Series): Summary of top peaks and SMSR.
        timing_info (dict): Time taken for each step.
    """
    t_start = time.time()
    timing = {}

    t0 = time.time()
    df_sorted = df_group.sort_values("Wavelength")
    timing["sort"] = time.time() - t0

    t0 = time.time()
    intensities = df_sorted["Intensity"].values
    dB_intensities = df_sorted["dB_Intensity"].values
    wavelengths = df_sorted["Wavelength"].values
    peak_indices, _ = find_peaks(dB_intensities)
    timing["find_peaks"] = time.time() - t0

    t0 = time.time()
    if len(peak_indices) == 0:
        peak_series = pd.Series(
            {
                "highest_peak_wavelength": np.nan,
                "highest_peak_intensity_linear": np.nan,
                "second_peak_wavelength": np.nan,
                "second_peak_intensity_linear": np.nan,
                "SMSR_dB": np.nan,
                "SMSR_linear": np.nan,
            }
        )
        timing["ordering"] = 0.0
        timing["extraction"] = 0.0
        return peak_series, timing

    sorted_order = np.argsort(dB_intensities[peak_indices])[::-1]
    timing["ordering"] = time.time() - t0

    t0 = time.time()
    highest_idx = peak_indices[sorted_order[0]]
    highest_peak_wavelength = wavelengths[highest_idx]
    highest_peak_intensity_linear = intensities[highest_idx]

    if len(sorted_order) > 1:
        second_idx = peak_indices[sorted_order[1]]
        second_peak_wavelength = wavelengths[second_idx]
        second_peak_intensity_linear = intensities[second_idx]
        second_peak_dB = dB_intensities[second_idx]

        SMSR_dB = -second_peak_dB
        SMSR_linear = highest_peak_intensity_linear / second_peak_intensity_linear
    else:
        second_peak_wavelength = np.nan
        second_peak_intensity_linear = np.nan
        SMSR_dB = np.nan
        SMSR_linear = np.nan

    peak_series = pd.Series(
        {
            "highest_peak_wavelength": highest_peak_wavelength,
            "highest_peak_intensity_linear": highest_peak_intensity_linear,
            "second_peak_wavelength": second_peak_wavelength,
            "second_peak_intensity_linear": second_peak_intensity_linear,
            "SMSR_dB": SMSR_dB,
            "SMSR_linear": SMSR_linear,
        }
    )
    timing["extraction"] = time.time() - t0

    return peak_series, timing


def process_export_and_peaks(filepath, wafer_code, decoder_df):
    print(f"\n=== Starting processing for {wafer_code} ===")
    total_t0 = time.time()

    spectra_output_path = EXPORTS_FILE_PATH / f"{ANALYSIS_RUN_NAME}_{wafer_code}_spectra_formatted.csv"
    peak_output_path = EXPORTS_FILE_PATH / f"{ANALYSIS_RUN_NAME}_{wafer_code}_peaks_summary.csv"

    accumulator = {}
    data_point_count = {}
    chunk_counter = 0

    spectra_columns = ["TYPE", "TE_LABEL", "Wavelength", "Intensity", "dB_Intensity"]
    peak_columns = [
        "Highest Peak (Wavelength)",
        "Highest Peak (Linear Intensity)",
        "Second Peak (Wavelength)",
        "Second Peak (Linear Intensity)",
        "SMSR_dB",
        "SMSR_linear",
        "TE_LABEL",
    ]

    with open(spectra_output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(spectra_columns)
    with open(peak_output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(peak_columns)

    completed_labels = 0
    spectra_buffer = []
    peak_buffer = []

    for chunk, data_points_threshold, base_time in transform_raw_file(filepath, wafer_code, decoder_df):
        chunk_counter += 1
        chunk_start = time.time()

        t_peaks_breakdown = {}
        t_peak_total = 0
        t_actual_write_total = 0
        t_dB_total = 0

        for te_label, group in chunk.groupby("TE_LABEL"):
            if te_label not in accumulator:
                accumulator[te_label] = [group]
                data_point_count[te_label] = len(group)
            else:
                accumulator[te_label].append(group)
                data_point_count[te_label] += len(group)

            if data_point_count[te_label] >= data_points_threshold:
                full_data = pd.concat(accumulator[te_label], ignore_index=True)

                # dB calculation
                t_dB_start = time.time()
                max_intensity = full_data["Intensity"].max()
                safe_intensity = np.where(full_data["Intensity"] > 0, full_data["Intensity"], np.nan)
                full_data["dB_Intensity"] = 10 * np.log10(safe_intensity / max_intensity)
                t_dB_end = time.time()
                t_dB_total += t_dB_end - t_dB_start

                # Peak extraction
                t_peak_start = time.time()
                peak_series, peak_times = extract_top_two_peaks(full_data)
                t_peak_end = time.time()
                t_peak_total += t_peak_end - t_peak_start

                for k, v in peak_times.items():
                    t_peaks_breakdown[k] = t_peaks_breakdown.get(k, 0.0) + v

                peak_series["TE_LABEL"] = te_label
                peak_buffer.append(peak_series)
                spectra_buffer.append(full_data)

                completed_labels += 1
                del accumulator[te_label]
                del data_point_count[te_label]

                # Flush if 1000 TE_LABELs completed
                if completed_labels >= 1000:
                    t_actual_write_start = time.time()
                    pd.concat(spectra_buffer).to_csv(spectra_output_path, mode="a", header=False, index=False)
                    pd.DataFrame(peak_buffer).to_csv(peak_output_path, mode="a", header=False, index=False)
                    t_actual_write_end = time.time()
                    t_actual_write_total += t_actual_write_end - t_actual_write_start

                    spectra_buffer.clear()
                    peak_buffer.clear()
                    completed_labels = 0

        chunk_total = time.time() - chunk_start
        print(f"Chunk {chunk_counter} Summary:")
        print(f"  Base transform: {base_time:.2f}s")
        print(f"  dB Calculation: {t_dB_total:.2f}s")
        print(f"  Peak Calculation Total: {t_peak_total:.2f}s")
        print(f"  Peak detection breakdown:")
        for step, t in t_peaks_breakdown.items():
            print(f"    {step:>10}: {t:.2f}s")
        print(f"  Actual writing time: {t_actual_write_total:.2f}s")
        print(f"  Chunk total:    {chunk_total:.2f}s\n")

    # Final flush
    if spectra_buffer:
        pd.concat(spectra_buffer).to_csv(spectra_output_path, mode="a", header=False, index=False)
    if peak_buffer:
        pd.DataFrame(peak_buffer).to_csv(peak_output_path, mode="a", header=False, index=False)

    print(f"=== Completed processing {wafer_code} in {time.time() - total_t0:.2f} seconds ===")


# ------------------------------------- Watchdog Calling Code ------------------------------------ #
monitored_folder = ROOT_DIR / "monitored_folder"
log_path = monitored_folder / "detection_log.txt"

print(f"# --------------------------- LIV Automatic Spectra Analyser (Vadan Khan) v1.1 -------------------------- #")
print("(Do not close this command window)")
print(f"\nWatching folder: {monitored_folder}")


# Updated wafer code extractor from LIV CSV filename
def extract_testinfo_from_liv(folder_path):
    for file in Path(folder_path).iterdir():
        if file.name.startswith("LIV_") and file.suffix == ".csv":
            parts = file.name.split("_")
            if len(parts) >= 3:
                tool_name = parts[0] + "_" + parts[1]  # e.g., LIV_53
                wafer_code = parts[2]  # e.g., QCI44

                # Search for COD variants in the entire filename
                filename_upper = file.name.upper()
                if "COD250" in filename_upper:
                    test_type = "COD250"
                elif "COD70" in filename_upper:
                    test_type = "COD70"
                elif "COD" in filename_upper:
                    test_type = "COD"
                else:
                    test_type = "UNKNOWN"

                return tool_name, wafer_code, test_type
    return None, None, None


# Combined wait: wait for file to appear and become readable, MAX WAIT 300s
def wait_for_file_to_appear_and_be_readable(filepath, max_wait=300, delay=1):
    wait_time = 0
    while not filepath.exists() and wait_time < max_wait:
        print(f"Waiting for {filepath.name} to appear... ({wait_time}s elapsed)")
        time.sleep(delay)
        wait_time += delay

    if not filepath.exists():
        return False

    for attempt in range(max_wait):
        try:
            with open(filepath, "rb"):
                return True
        except (PermissionError, FileNotFoundError):
            print(f"Waiting for {filepath.name} to be ready... ({attempt}s elapsed)")
            time.sleep(delay)
    return False


def initialise_spectra_processing(wafer_code, detection_time, file_path):
    if wafer_code:
        product_code = wafer_code[:2]
        print(f"Extracted wafer code: {wafer_code}")

        if product_code == "QC":
            decoder_path = ROOT_DIR / "decoders" / SUBARU_DECODER
        elif product_code in ("QD", "NV"):
            decoder_path = ROOT_DIR / "decoders" / HALO_DECODER
        else:
            message = f"Unsupported product code: {product_code}"
            print(message)
            with open(log_path, "a", encoding="utf-8") as log_file:
                log_file.write(f"[{detection_time}] {message}\n")
            return

        decoder_df = load_decoder(decoder_path)
        process_export_and_peaks(file_path, wafer_code, decoder_df)  # Main Spectra Processor
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"[{detection_time}] ✅ Processed successfully: {file_path.name} (Wafer: {wafer_code})\n")
    else:
        message = f"No valid wafer code found in filename: {file_path.name}"
        print(message)
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"[{detection_time}] ❌ {message}\n")


# Handler for new **folders**
class WaferFileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory:
            return

        def analysis_job():
            folder_path = Path(event.src_path)
            detection_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            with open(log_path, "a", encoding="utf-8") as log_file:
                log_file.write(f"[{detection_time}] Detected new folder: {folder_path.name}\n")

            print(f"\nDetected new wafer folder: {folder_path.name}")

            raw_csv_path = folder_path / "0_LIV_Pulse_Interval_Opt" / "Raw.csv"

            if not wait_for_file_to_appear_and_be_readable(
                raw_csv_path
            ):  # Function that Waits to only read when file fully copied over
                message = f"Raw.csv did not appear or never became readable in: {folder_path.name}"
                print(message)
                with open(log_path, "a", encoding="utf-8") as log_file:
                    log_file.write(f"[{detection_time}] ❌ {message}\n")
                return

            tool_name, wafer_code, test_type = extract_testinfo_from_liv(folder_path)
            print(f"Detected Tool: {tool_name}, Wafer Code: {wafer_code}, Test Type: {test_type}")

            if test_type in ("COD70", "COD250"):
                message = f"Skipping analysis for test type {test_type} in: {folder_path.name}"
                print(message)
                with open(log_path, "a", encoding="utf-8") as log_file:
                    log_file.write(f"[{detection_time}] ⚠️ {message}\n")
                print(f"\n\nWatching folder: {monitored_folder}")
                return

            initialise_spectra_processing(wafer_code, detection_time, raw_csv_path)
            print(f"\n\nWatching folder: {monitored_folder}")

        # Run the job in a background thread so Ctrl+C can still work
        threading.Thread(target=analysis_job, daemon=True).start()


# Watchdog setup
observer = Observer()
event_handler = WaferFileHandler()
observer.schedule(event_handler, str(monitored_folder), recursive=False)
observer.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    observer.stop()
observer.join()
