#!/usr/bin/env python3
"""
Create a lightweight GEBCO NetCDF by subsampling a global GEBCO grid.

Recommended source file:
  GEBCO_2025 Grid (ice surface elevation) -> global coverage -> netCDF
  https://www.gebco.net/data-products/gridded-bathymetry-data

Example:
  python3 make_lightweight_gebco.py \
    --input /path/to/GEBCO_2025.nc \
    --output /path/to/GEBCO_2025_6min.nc \
    --arc-min 6
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import netcdf_file


def pick_name(var_names: list[str], candidates: list[str]) -> str | None:
    lower_map = {name.lower(): name for name in var_names}
    for candidate in candidates:
        if candidate in lower_map:
            return lower_map[candidate]
    return None


def detect_variables(nc) -> tuple[str, str, str]:
    var_names = list(nc.variables.keys())
    lon_name = pick_name(var_names, ["lon", "longitude", "x"])
    lat_name = pick_name(var_names, ["lat", "latitude", "y"])
    height_name = pick_name(var_names, ["elevation", "height", "z", "depth", "bathymetry"])

    if not all([lon_name, lat_name, height_name]):
        raise ValueError(f"Could not detect lon/lat/height variables from: {var_names}")

    return lon_name, lat_name, height_name


def calc_stride(arc_min: float, native_arc_sec: float = 15.0) -> int:
    target_arc_sec = arc_min * 60.0
    stride = target_arc_sec / native_arc_sec
    if abs(stride - round(stride)) > 1e-9:
        raise ValueError("Target resolution must be a multiple of 15 arc-seconds.")
    return int(round(stride))


def copy_attrs(src_var, dst_var) -> None:
    attrs = getattr(src_var, "_attributes", {})
    for key, value in attrs.items():
        try:
            setattr(dst_var, key, value)
        except Exception:
            pass


def prepare_readable_netcdf3(input_path: Path) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    """
    Convert a NetCDF4 GEBCO file into a temporary NetCDF3 classic file via nccopy,
    so scipy.io.netcdf_file can read it without extra Python dependencies.
    """
    nccopy = shutil.which("nccopy")
    if nccopy is None:
        raise RuntimeError("nccopy was not found. Please install netCDF command-line tools.")

    tmpdir = tempfile.TemporaryDirectory()
    converted = Path(tmpdir.name) / "converted.nc"
    subprocess.run(
        [nccopy, "-k", "classic", str(input_path), str(converted)],
        check=True,
    )
    return converted, tmpdir


def main() -> None:
    parser = argparse.ArgumentParser(description="Subsample a global GEBCO NetCDF into a lightweight grid.")
    parser.add_argument("--input", required=True, help="Path to the source GEBCO NetCDF.")
    parser.add_argument("--output", required=True, help="Path to the output lightweight NetCDF.")
    parser.add_argument("--arc-min", type=float, default=6.0, help="Target grid spacing in arc-minutes.")
    parser.add_argument("--row-chunk", type=int, default=512, help="Rows per write chunk before subsampling.")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stride = calc_stride(args.arc_min)

    readable_path, tmpdir = prepare_readable_netcdf3(input_path)
    try:
        with netcdf_file(str(readable_path), "r", mmap=True) as src:
            lon_name, lat_name, height_name = detect_variables(src)

            lon_src = src.variables[lon_name]
            lat_src = src.variables[lat_name]
            height_src = src.variables[height_name]

            lon_out = np.array(lon_src[::stride], dtype=np.float32)
            lat_out = np.array(lat_src[::stride], dtype=np.float32)

            with netcdf_file(str(output_path), "w") as dst:
                dst.history = (
                    f"Subsampled from {input_path.name} to {args.arc_min} arc-min spacing "
                    f"using stride={stride} with make_lightweight_gebco.py"
                )

                dst.createDimension("lon", len(lon_out))
                dst.createDimension("lat", len(lat_out))

                lon_var = dst.createVariable("lon", "f4", ("lon",))
                lat_var = dst.createVariable("lat", "f4", ("lat",))
                elev_var = dst.createVariable("Height", "i2", ("lat", "lon"))

                lon_var[:] = lon_out
                lat_var[:] = lat_out

                copy_attrs(lon_src, lon_var)
                copy_attrs(lat_src, lat_var)
                copy_attrs(height_src, elev_var)

                src_rows = height_src.shape[0]
                out_row = 0
                for row_start in range(0, src_rows, args.row_chunk * stride):
                    row_stop = min(src_rows, row_start + args.row_chunk * stride)
                    block = np.array(height_src[row_start:row_stop:stride, ::stride])
                    rows_written = block.shape[0]
                    elev_var[out_row:out_row + rows_written, :] = block
                    out_row += rows_written
    finally:
        if tmpdir is not None:
            tmpdir.cleanup()

    print(f"Created: {output_path}")
    print(f"Resolution: {args.arc_min} arc-min")


if __name__ == "__main__":
    main()
