import os
import numpy as np
import xarray as xr
from enum import Enum
from datetime import datetime, timedelta
from fastapi import HTTPException


class CosmoForecast(str, Enum):
    VNXQ94 = "VNXQ94"
    VNXZ32 = "VNXZ32"


def verify_cosmo_area_forecast(model, variables, forecast_date, ll_lat, ll_lng, ur_lat, ur_lng):
    return True


def verify_cosmo_point_forecast(model, variables, forecast_date, lat, lng):
    return True


def get_cosmo_area_forecast(filesystem, model, variables, forecast_date, ll_lat, ll_lng, ur_lat, ur_lng):
    file = os.path.join(filesystem, "media/meteoswiss/cosmo", model, "{}.{}0000.nc".format(model, forecast_date))
    if not os.path.isfile(file):
        raise HTTPException(status_code=400, detail="Data not available for COSMO {} for the following date: {}".format(model, forecast_date))
    output = {}
    with xr.open_mfdataset(file) as ds:
        bad_variables = []
        for var in variables:
            if var not in ds.variables.keys():
                bad_variables.append(var)
        if len(bad_variables) > 0:
            raise HTTPException(status_code=400,
                                detail="{} are bad variables for COSMO {}. Please select from: {}".format(
                                    ", ".join(bad_variables), model, ", ".join(ds.keys())))

        output["time"] = np.array(ds.variables["time"].values, dtype=str).tolist()
        x, y = np.where(((ds.variables["lat_1"] >= ll_lat) & (ds.variables["lat_1"] <= ur_lat) & (ds.variables["lon_1"] >= ll_lng) & (ds.variables["lon_1"] <= ur_lng)))

        if len(x) == 0:
            raise HTTPException(status_code=400,
                                detail="Data not available for COSMO {} for the requsted area.".format(model))

        x_min, x_max, y_min, y_max = min(x), max(x), min(y), max(y)
        output["lat"] = ds.variables["lat_1"][x_min:x_max, y_min:y_max].values.tolist()
        output["lng"] = ds.variables["lon_1"][x_min:x_max, y_min:y_max].values.tolist()
        for var in variables:
            if var in ds.variables.keys():
                if len(ds.variables[var].dims) == 3:
                    data = ds.variables[var][:, x_min:x_max, y_min:y_max].values
                elif len(ds.variables[var].dims) == 4:
                    data = ds.variables[var][:, 0, x_min:x_max, y_min:y_max].values
                elif len(ds.variables[var].dims) == 5:
                    data = ds.variables[var][:, 0, 0, x_min:x_max, y_min:y_max].values
                else:
                    data = []
                output[var] = {"name": ds.variables[var].attrs["long_name"],
                               "unit": ds.variables[var].attrs["units"],
                               "data": np.where(np.isnan(data), None, data).tolist()}
            else:
                output[var] = []
    return output


def get_cosmo_point_forecast(filesystem, model, variables, forecast_date, lat, lng):
    file = os.path.join(filesystem, "media/meteoswiss/cosmo", model, "{}.{}0000.nc".format(model, forecast_date))
    if not os.path.isfile(file):
        raise HTTPException(status_code=400, detail="Data not available for COSMO {} for the following date: {}".format(model, forecast_date))
    output = {}
    with xr.open_mfdataset(file) as ds:
        bad_variables = []
        for var in variables:
            if var not in ds.variables.keys():
                bad_variables.append(var)
        if len(bad_variables) > 0:
            raise HTTPException(status_code=400,
                                detail="{} are bad variables for COSMO {}. Please select from: {}".format(
                                    ", ".join(bad_variables), model, ", ".join(ds.keys())))

        output["time"] = np.array(ds.variables["time"].values, dtype=str).tolist()

        dist = ((ds.variables["lat_1"] - lat) ** 2 + (ds.variables["lon_1"] - lng) ** 2) ** 0.5
        xy = np.unravel_index(dist.argmin(), dist.shape)
        x, y = xy[0], xy[1]
        output["lat"] = float(ds.variables["lat_1"][x, y].values)
        output["lng"] = float(ds.variables["lon_1"][x, y].values)

        for var in variables:
            if var in ds.variables.keys():
                if len(ds.variables[var].dims) == 3:
                    data = ds.variables[var][:, x, y].values
                elif len(ds.variables[var].dims) == 4:
                    data = ds.variables[var][:, 0, x, y].values
                elif len(ds.variables[var].dims) == 5:
                    data = ds.variables[var][:, 0, 0, x, y].values
                else:
                    data = []
                output[var] = {"name": ds.variables[var].attrs["long_name"],
                               "unit": ds.variables[var].attrs["units"],
                               "data": np.where(np.isnan(data), None, data).tolist()}
            else:
                output[var] = []
    return output


class CosmoReanalysis(str, Enum):
    VNJK21 = "VNJK21"
    VNXQ34 = "VNXQ34"


def verify_cosmo_area_reanalysis(model, variables, start_time, end_time, ll_lat, ll_lng, ur_lat, ur_lng):
    return True


def verify_cosmo_point_reanalysis(model, variables, start_time, end_time, lat, lng):
    return True


def get_cosmo_area_reanalysis(filesystem, model, variables, start_date, end_date, ll_lat, ll_lng, ur_lat, ur_lng):
    # For reanalysis files the date on the file is one day after the data in the file
    folder = os.path.join(filesystem, "media/meteoswiss/cosmo", model)
    start_date = datetime.strptime(start_date, '%Y%m%d')
    end_date = datetime.strptime(end_date, '%Y%m%d')
    files = [os.path.join(folder, "{}.{}0000.nc".format(model, (start_date+timedelta(days=x)).strftime("%Y%m%d")))
             for x in range(1, (end_date-start_date).days + 2)]
    bad_files = []
    for file in files:
        if not os.path.isfile(file):
            bad_files.append(file.split("/")[-1].split(".")[1][:8])
    if len(bad_files) > 0:
        raise HTTPException(status_code=400, detail="Data not available for COSMO {} for the following dates: {}".format(model, ", ".join(bad_files)))
    output = {}
    with xr.open_mfdataset(files) as ds:
        bad_variables = []
        for var in variables:
            if var not in ds.variables.keys():
                bad_variables.append(var)
        if len(bad_variables) > 0:
            raise HTTPException(status_code=400, detail="{} are bad variables for COSMO {}. Please select from: {}".format(", ".join(bad_variables), model, ", ".join(ds.keys())))
        output["time"] = np.array(ds.variables["time"].values, dtype=str).tolist()
        x, y = np.where(((ds.variables["lat_1"] >= ll_lat) & (ds.variables["lat_1"] <= ur_lat) & (
                    ds.variables["lon_1"] >= ll_lng) & (ds.variables["lon_1"] <= ur_lng)))

        if len(x) == 0:
            raise HTTPException(status_code=400,
                                detail="Data not available for COSMO {} for the requsted area.".format(model))

        x_min, x_max, y_min, y_max = min(x), max(x), min(y), max(y)
        output["lat"] = ds.variables["lat_1"][x_min:x_max, y_min:y_max].values.tolist()
        output["lng"] = ds.variables["lon_1"][x_min:x_max, y_min:y_max].values.tolist()
        for var in variables:
            if var in ds.variables.keys():
                if len(ds.variables[var].dims) == 3:
                    data = ds.variables[var][:, x_min:x_max, y_min:y_max].values
                elif len(ds.variables[var].dims) == 4:
                    data = ds.variables[var][:, 0, x_min:x_max, y_min:y_max].values
                else:
                    data = []
                output[var] = {"name": ds.variables[var].attrs["long_name"],
                               "unit": ds.variables[var].attrs["units"],
                               "data": np.where(np.isnan(data), None, data).tolist()}
            else:
                output[var] = []
    return output


def get_cosmo_point_reanalysis(filesystem, model, variables, start_date, end_date, lat, lng):
    # For reanalysis files the date on the file is one day after the data in the file
    folder = os.path.join(filesystem, "media/meteoswiss/cosmo", model)
    start_date = datetime.strptime(start_date, '%Y%m%d')
    end_date = datetime.strptime(end_date, '%Y%m%d')
    files = [os.path.join(folder, "{}.{}0000.nc".format(model, (start_date+timedelta(days=x)).strftime("%Y%m%d")))
             for x in range(1, (end_date-start_date).days + 2)]
    bad_files = []
    for file in files:
        if not os.path.isfile(file):
            bad_files.append(file.split("/")[-1].split(".")[1][:8])
    if len(bad_files) > 0:
        raise HTTPException(status_code=400, detail="Data not available for COSMO {} for the following dates: {}".format(model, ", ".join(bad_files)))
    output = {}
    with xr.open_mfdataset(files) as ds:
        bad_variables = []
        for var in variables:
            if var not in ds.variables.keys():
                bad_variables.append(var)
        if len(bad_variables) > 0:
            raise HTTPException(status_code=400, detail="{} are bad variables for COSMO {}. Please select from: {}".format(", ".join(bad_variables), model, ", ".join(ds.keys())))
        output["time"] = np.array(ds.variables["time"].values, dtype=str).tolist()

        dist = ((ds.variables["lat_1"] - lat)**2 + (ds.variables["lon_1"] - lng)**2)**0.5
        xy = np.unravel_index(dist.argmin(), dist.shape)
        x, y = xy[0], xy[1]
        output["lat"] = float(ds.variables["lat_1"][x, y].values)
        output["lng"] = float(ds.variables["lon_1"][x, y].values)

        for var in variables:
            if var in ds.variables.keys():
                if len(ds.variables[var].dims) == 3:
                    data = ds.variables[var][:, x, y].values
                elif len(ds.variables[var].dims) == 4:
                    data = ds.variables[var][:, 0, x, y].values
                else:
                    data = []
                output[var] = {"name": ds.variables[var].attrs["long_name"],
                               "unit": ds.variables[var].attrs["units"],
                               "data": np.where(np.isnan(data), None, data).tolist()}
            else:
                output[var] = []
    return output
