import os
import netCDF4
import xarray as xr
from enum import Enum


class CosmoForecast(str, Enum):
    VNXQ94 = "VNXQ94"
    VNXZ32 = "VNXZ32"


def verify_cosmo_forecast(model, forecast_date, ll_lat, ll_lng, ur_lat, ur_lng):
    return True


def get_cosmo_forecast(model, forecast_date, ll_lat, ll_lng, ur_lat, ur_lng):
    folder = os.path.join("../../filesystem/media/meteoswiss/cosmo", model)
    ds = xr.open_mfdataset(os.path.join(folder, "{}.{}0000.nc".format(model, forecast_date)))
    print(ds)
    return "Getting data"


class CosmoReanalysis(str, Enum):
    VNJK21 = "VNJK21"
    VNXQ34 = "VNXQ34"


def verify_cosmo_reanalysis(model, start_time, end_time, ll_lat, ll_lng, ur_lat, ur_lng):
    return True


def get_cosmo_reanalysis(model, start_time, end_time, ll_lat, ll_lng, ur_lat, ur_lng):
    return "Getting data"
