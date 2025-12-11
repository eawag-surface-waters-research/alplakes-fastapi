import os
import json
import netCDF4
import numpy as np
import xarray as xr
import pandas as pd
from enum import Enum
from typing import Dict, List, Union, Any
from pydantic import BaseModel, field_validator
from fastapi import HTTPException
from fastapi.responses import FileResponse
from datetime import datetime, timedelta, timezone, date
from dateutil.relativedelta import relativedelta, SU

from app import functions

class MetadataLake(BaseModel):
    name: str
    depth: List[float]
    start_date: date
    end_date: date
    missing_dates: List[date]
    height: int
    width: int

class Metadata(BaseModel):
    model: str
    lakes: List[MetadataLake]

class ResponseModelPoint(BaseModel):
    time: List[datetime]
    lat: float
    lng: float
    depth: functions.VariableKeyModel1D
    distance: functions.VariableKeyModel1D
    variables: Dict[str, functions.VariableKeyModel1D]
    @field_validator('time')
    @classmethod
    def validate_timezone(cls, value):
        if isinstance(value, list):
            for v in value:
                if v.tzinfo is None:
                    raise ValueError('time must have a timezone')
        elif value.tzinfo is None:
            raise ValueError('time must have a timezone')
        return value


class ResponseModelLayer(BaseModel):
    time: datetime
    lat: List[List[Any]]
    lng: List[List[Any]]
    depth: functions.VariableKeyModel1D
    variables: Dict[str, functions.VariableKeyModel2D]
    @field_validator('time')
    @classmethod
    def validate_timezone(cls, value):
        if isinstance(value, list):
            for v in value:
                if v.tzinfo is None:
                    raise ValueError('time must have a timezone')
        elif value.tzinfo is None:
            raise ValueError('time must have a timezone')
        return value


class ResponseModelAverageLayer(BaseModel):
    time: List[datetime]
    depth: functions.VariableKeyModel1D
    variable: functions.VariableKeyModel1D
    @field_validator('time')
    @classmethod
    def validate_timezone(cls, value):
        if isinstance(value, list):
            for v in value:
                if v.tzinfo is None:
                    raise ValueError('time must have a timezone')
        elif value.tzinfo is None:
            raise ValueError('time must have a timezone')
        return value


class ResponseModelAverageBottom(BaseModel):
    lat: List[List[Any]]
    lng: List[List[Any]]
    variable: functions.VariableKeyModel1D


class ResponseModelProfile(BaseModel):
    time: datetime
    lat: float
    lng: float
    depth: functions.VariableKeyModel1D
    variables: Dict[str, functions.VariableKeyModel1D]
    @field_validator('time')
    @classmethod
    def validate_timezone(cls, value):
        if isinstance(value, list):
            for v in value:
                if v.tzinfo is None:
                    raise ValueError('time must have a timezone')
        elif value.tzinfo is None:
            raise ValueError('time must have a timezone')
        return value

class ResponseModelDepthTime(BaseModel):
    time: List[datetime]
    lat: float
    lng: float
    depth: functions.VariableKeyModel1D
    distance: functions.VariableKeyModel1D
    variables: Dict[str, functions.VariableKeyModel1D]
    @field_validator('time')
    @classmethod
    def validate_timezone(cls, value):
        if isinstance(value, list):
            for v in value:
                if v.tzinfo is None:
                    raise ValueError('time must have a timezone')
        elif value.tzinfo is None:
            raise ValueError('time must have a timezone')
        return value

class ResponseModelTransect(BaseModel):
    time: datetime
    lat: List[float]
    lng: List[float]
    depth: functions.VariableKeyModel1D
    distance: functions.VariableKeyModel1D
    variables: Dict[str, functions.VariableKeyModel1D]
    @field_validator('time')
    @classmethod
    def validate_timezone(cls, value):
        if isinstance(value, list):
            for v in value:
                if v.tzinfo is None:
                    raise ValueError('time must have a timezone')
        elif value.tzinfo is None:
            raise ValueError('time must have a timezone')
        return value

class ResponseModelTransectPeriod(BaseModel):
    time: List[datetime]
    lat: List[float]
    lng: List[float]
    depth: functions.VariableKeyModel1D
    distance: functions.VariableKeyModel1D
    variables: Dict[str, functions.VariableKeyModel1D]
    @field_validator('time')
    @classmethod
    def validate_timezone(cls, value):
        if isinstance(value, list):
            for v in value:
                if v.tzinfo is None:
                    raise ValueError('time must have a timezone')
        elif value.tzinfo is None:
            raise ValueError('time must have a timezone')
        return value


def get_metadata(filesystem):
    metadata = []
    models = [m for m in os.listdir(os.path.join(filesystem, "media/simulations")) if m in ["delft3d-flow", "mitgcm"]]
    for model in models:
        lakes = os.listdir(os.path.join(filesystem, "media/simulations", model, "results"))
        m = {"model": model, "lakes": []}
        for lake in lakes:
            try:
                m["lakes"].append(get_metadata_lake(filesystem, model, lake))
            except:
                print("Failed for {}".format(lake))
        metadata.append(m)
    return metadata


def get_metadata_lake(filesystem, model, lake):
    path = os.path.join(os.path.join(filesystem, "media/simulations", model, "results", lake))
    if not os.path.isdir(path):
        raise HTTPException(status_code=400,
                            detail="{} simulation results are not available for {}"
                            .format(model, lake))
    files = os.listdir(path)
    files = [file for file in files if len(file.split(".")[0]) == 8 and file.split(".")[1] == "nc"]
    files.sort()
    combined = '_'.join(files)
    missing_dates = []

    if model == "delft3d-flow":
        with netCDF4.Dataset(os.path.join(path, files[0])) as nc:
            height = len(nc.dimensions["M"])
            width = len(nc.dimensions["N"])
            start_date = functions.convert_from_unit(nc.variables["time"][0], nc.variables["time"].units)

        with netCDF4.Dataset(os.path.join(path, files[-1])) as nc:
            end_date = functions.convert_from_unit(nc.variables["time"][-1], nc.variables["time"].units)
            depths = np.array(nc.variables["ZK_LYR"][:]) * -1
            depths = depths[depths > 0]
            depths.sort()
            depths = [float("%.2f" % d) for d in depths]
    elif model == "mitgcm":
        with netCDF4.Dataset(os.path.join(path, files[0])) as nc:
            height = len(nc.dimensions["Y"])
            width = len(nc.dimensions["X"])
            start_date = functions.convert_from_unit(nc.variables["time"][0], nc.variables["time"].units)

        with netCDF4.Dataset(os.path.join(path, files[-1])) as nc:
            end_date = functions.convert_from_unit(nc.variables["time"][-1], nc.variables["time"].units)
            depths = np.array(nc.variables["depth"][:])
            depths = depths[depths > 0]
            depths.sort()
            depths = [float("%.2f" % d) for d in depths]
    else:
        raise ValueError("Model not recognised.")

    for d in functions.daterange(start_date, end_date, days=7):
        if d.strftime('%Y%m%d') not in combined:
            missing_dates.append(d.strftime("%Y-%m-%d"))

    return {"name": lake,
            "depth": depths,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "missing_dates": missing_dates,
            "height": height,
            "width": width}


class Models(str, Enum):
    delft3dflow = "delft3d-flow"
    mitgcm = "mitgcm"


class Variables(str, Enum):
    temperature = "temperature"
    velocity = "velocity"
    geometry = "geometry"
    thermocline = "thermocline"


def get_simulations_point(filesystem, model, lake, start, end, depth, latitude, longitude, variables):
    if model == "delft3d-flow":
        return get_simulations_point_delft3dflow(filesystem, lake, start, end, depth, latitude, longitude, variables)
    elif model == "mitgcm":
        return get_simulations_point_mitgcm(filesystem, lake, start, end, depth, latitude, longitude, variables)
    else:
        raise HTTPException(status_code=400,
                            detail="Apologies profile extraction not available for {}".format(model))


def get_simulations_point_delft3dflow(filesystem, lake, start, end, depth, latitude, longitude, variables, nodata=-999.0):
    model = "delft3d-flow"
    lakes = os.path.join(filesystem, "media/simulations", model, "results")
    variables = [v.lower() for v in variables]
    if not os.path.isdir(os.path.join(lakes, lake)):
        raise HTTPException(status_code=400,
                            detail="{} simulation results are not available for {} please select from: [{}]"
                            .format(model, lake, ", ".join(os.listdir(lakes))))
    weeks = functions.sundays_between_dates(datetime.strptime(start[0:8], "%Y%m%d").replace(tzinfo=timezone.utc),
                                            datetime.strptime(end[0:8], "%Y%m%d").replace(tzinfo=timezone.utc))
    files = [os.path.join(lakes, lake, "{}.nc".format(week.strftime("%Y%m%d"))) for week in weeks]

    files = [file for file in files if os.path.isfile(file)]
    if len(files) == 0:
        raise HTTPException(status_code=400, detail="Apologies data is not available for your requested period")

    start_datetime = datetime.strptime(start, "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
    end_datetime = datetime.strptime(end, "%Y%m%d%H%M").replace(tzinfo=timezone.utc)

    with xr.open_mfdataset(files, data_vars='minimal', compat='override', coords='minimal') as ds:
        ds['time'] = ds.indexes['time'].tz_localize('UTC')
        ds = ds.sel(time=slice(start_datetime, end_datetime))
        if len(ds['time']) == 0:
            raise HTTPException(status_code=400,
                                detail="No timesteps available between {} and {}".format(start, end))
        z = ds.ZK_LYR[0, :].values * -1 if len(ds.ZK_LYR.shape) == 2 else ds.ZK_LYR[:].values * -1
        depth_index = functions.get_closest_index(depth, z)
        depth = float(z[depth_index])
        lat_grid, lng_grid = functions.coordinates_to_latlng(ds.XZ[:].values, ds.YZ[:].values)
        x_index, y_index, distance = functions.get_closest_location(latitude, longitude, lat_grid, lng_grid)
        time = functions.alplakes_time(ds.time.values, "nano")
        output = {"time": time,
                  "lat": lat_grid[x_index, y_index],
                  "lng": lng_grid[x_index, y_index],
                  "distance": {"data": distance, "unit": "m",
                               "description": "Distance from requested location to center of closest grid point"},
                  "depth": {"data": depth, "unit": "m",
                            "description": "Distance from the surface to the closest grid point to requested depth"},
                  "variables": {}
                  }
        if "temperature" in variables:
            t = ds.R1.isel(M=x_index, N=y_index, KMAXOUT_RESTR=depth_index, LSTSCI=0).values
            output["variables"]["temperature"] = {"data": functions.filter_variable(t), "unit": "degC", "description": "Water temperature"}
        if "velocity" in variables:
            u, v, = functions.rotate_velocity(
                ds.U1.isel(MC=x_index, N=y_index, KMAXOUT_RESTR=depth_index).values,
                ds.V1.isel(M=x_index, NC=y_index, KMAXOUT_RESTR=depth_index).values,
                ds.ALFAS.isel(M=x_index, N=y_index).values)
            output["variables"]["u"] = {"data": functions.filter_variable(u, decimals=5), "unit": "m/s", "description": "Eastward flow velocity"}
            output["variables"]["v"] = {"data": functions.filter_variable(v, decimals=5), "unit": "m/s", "description": "Northward flow velocity"}
    return output


def get_simulations_point_mitgcm(filesystem, lake, start, end, depth, latitude, longitude, variables, nodata=-999.0):
    model = "mitgcm"
    lakes = os.path.join(filesystem, "media/simulations", model, "results")
    variables = [v.lower() for v in variables]
    if not os.path.isdir(os.path.join(lakes, lake)):
        raise HTTPException(status_code=400,
                            detail="{} simulation results are not available for {} please select from: [{}]"
                            .format(model, lake, ", ".join(os.listdir(lakes))))
    weeks = functions.sundays_between_dates(datetime.strptime(start[0:8], "%Y%m%d").replace(tzinfo=timezone.utc),
                                            datetime.strptime(end[0:8], "%Y%m%d").replace(tzinfo=timezone.utc))
    files = [os.path.join(lakes, lake, "{}.nc".format(week.strftime("%Y%m%d"))) for week in weeks]

    files = [file for file in files if os.path.isfile(file)]
    if len(files) == 0:
        raise HTTPException(status_code=400, detail="Apologies data is not available for your requested period")

    start_datetime = datetime.strptime(start, "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
    end_datetime = datetime.strptime(end, "%Y%m%d%H%M").replace(tzinfo=timezone.utc)

    with xr.open_mfdataset(files, data_vars='minimal', compat='override', coords='minimal') as ds:
        ds['time'] = ds.indexes['time'].tz_localize('UTC')
        ds = ds.sel(time=slice(start_datetime, end_datetime))
        if len(ds['time']) == 0:
            raise HTTPException(status_code=400,
                                detail="No timesteps available between {} and {}".format(start, end))
        z = ds.depth[0, :].values if len(ds.depth.shape) == 2 else ds.depth[:].values
        depth_index = functions.get_closest_index(depth, z)
        depth = float(z[depth_index])
        if len(ds.lat.shape) == 2:
            lat_grid, lng_grid = ds.lat.values, ds.lng.values
        else:
            lat_grid, lng_grid = ds.lat.isel(time=0).values, ds.lng.isel(time=0).values
        mask = np.isnan(ds.t.isel(time=0, depth=0).values)
        lat_grid[mask] = np.nan
        lng_grid[mask] = np.nan
        x_index, y_index, distance = functions.get_closest_location(latitude, longitude, lat_grid, lng_grid, yx=True)
        time = functions.alplakes_time(ds.time.values, "nano")
        output = {"time": time,
                  "lat": lat_grid[y_index, x_index],
                  "lng": lng_grid[y_index, x_index],
                  "distance": {"data": distance, "unit": "m",
                               "description": "Distance from requested location to center of closest grid point"},
                  "depth": {"data": depth, "unit": "m",
                            "description": "Distance from the surface to the closest grid point to requested depth"},
                  "variables": {}
                  }
        if "temperature" in variables:
            t = ds.t.isel(X=x_index, Y=y_index, depth=depth_index).values
            output["variables"]["temperature"] = {"data": functions.filter_variable(t), "unit": "degC", "description": "Water temperature"}
        if "velocity" in variables:
            u = ds.u.isel(X=x_index, Y=y_index, depth=depth_index).values
            v = ds.v.isel(X=x_index, Y=y_index, depth=depth_index).values
            output["variables"]["u"] = {"data": functions.filter_variable(u, decimals=5), "unit": "m/s", "description": "Eastward flow velocity"}
            output["variables"]["v"] = {"data": functions.filter_variable(v, decimals=5), "unit": "m/s", "description": "Northward flow velocity"}
    return output


def get_simulations_layer(filesystem, model, lake, time, depth, variables):
    if model == "delft3d-flow":
        return get_simulations_layer_delft3dflow(filesystem, lake, time, depth, variables)
    elif model == "mitgcm":
        return get_simulations_layer_mitgcm(filesystem, lake, time, depth, variables)
    else:
        raise HTTPException(status_code=400,
                            detail="Apologies data is not available for {}".format(model))


def get_simulations_layer_delft3dflow(filesystem, lake, time, depth, variables):
    model = "delft3d-flow"
    variables = [v.lower() for v in variables]
    origin = datetime.strptime(time, "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
    last_sunday = origin + relativedelta(weekday=SU(-1))
    previous_sunday = last_sunday - timedelta(days=7)
    lakes = os.path.join(filesystem, "media/simulations", model, "results")
    if not os.path.isdir(os.path.join(lakes, lake)):
        raise HTTPException(status_code=400,
                            detail="{} simulation results are not available for {} please select from: [{}]"
                            .format(model, lake, ", ".join(os.listdir(lakes))))
    if os.path.isfile(os.path.join(lakes, lake, "{}.nc".format(last_sunday.strftime("%Y%m%d")))):
        file = os.path.join(lakes, lake, "{}.nc".format(last_sunday.strftime("%Y%m%d")))
    elif os.path.isfile(os.path.join(lakes, lake, "{}.nc".format(previous_sunday.strftime("%Y%m%d")))):
        file = os.path.join(lakes, lake, "{}.nc".format(previous_sunday.strftime("%Y%m%d")))
    else:
        raise HTTPException(status_code=400,
                            detail="Apologies data is not available for {} at {}".format(lake, time))
    with netCDF4.Dataset(file) as nc:
        converted_time = functions.convert_to_unit(origin, nc.variables["time"].units)
        time_index = functions.get_closest_index(converted_time, np.array(nc.variables["time"][:]))
        depth_index = functions.get_closest_index(depth, np.array(nc.variables["ZK_LYR"][:]) * -1)
        time = nc.variables["time"][time_index].tolist()
        depth = nc.variables["ZK_LYR"][depth_index].tolist() * -1
        lat_grid, lng_grid = functions.coordinates_to_latlng(nc.variables["XZ"][:], nc.variables["YZ"][:])
        output = {"time": functions.alplakes_time(time, nc.variables["time"].units),
               "depth": {"description": "Distance from the surface to the closest grid point to requested depth",
                         "units": nc.variables["ZK_LYR"].units,
                         "data": depth},
               "lat": functions.filter_variable(lat_grid, decimals=5, nodata=np.nan),
               "lng": functions.filter_variable(lng_grid, decimals=5, nodata=np.nan),
               "variables": {}
               }
        if "temperature" in variables:
            t = functions.filter_variable(nc.variables["R1"][time_index, 0, depth_index, :])
            output["variables"]["temperature"] = {"data": functions.filter_variable(t), "unit": "degC",
                                                  "description": "Water temperature"}
        if "velocity" in variables:
            u, v, = functions.rotate_velocity(nc.variables["U1"][time_index, depth_index, :],
                                              nc.variables["V1"][time_index, depth_index, :],
                                              nc.variables["ALFAS"][:])
            output["variables"]["u"] = {"data": functions.filter_variable(u, decimals=5), "unit": "m/s",
                                        "description": "Eastward flow velocity"}
            output["variables"]["v"] = {"data": functions.filter_variable(v, decimals=5), "unit": "m/s",
                                        "description": "Northward flow velocity"}
    return output


def get_simulations_layer_mitgcm(filesystem, lake, time, depth, variables):
    model = "mitgcm"
    variables = [v.lower() for v in variables]
    origin = datetime.strptime(time, "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
    last_sunday = origin + relativedelta(weekday=SU(-1))
    previous_sunday = last_sunday - timedelta(days=7)
    lakes = os.path.join(filesystem, "media/simulations", model, "results")
    if not os.path.isdir(os.path.join(lakes, lake)):
        raise HTTPException(status_code=400,
                            detail="{} simulation results are not available for {} please select from: [{}]"
                            .format(model, lake, ", ".join(os.listdir(lakes))))
    if os.path.isfile(os.path.join(lakes, lake, "{}.nc".format(last_sunday.strftime("%Y%m%d")))):
        file = os.path.join(lakes, lake, "{}.nc".format(last_sunday.strftime("%Y%m%d")))
    elif os.path.isfile(os.path.join(lakes, lake, "{}.nc".format(previous_sunday.strftime("%Y%m%d")))):
        file = os.path.join(lakes, lake, "{}.nc".format(previous_sunday.strftime("%Y%m%d")))
    else:
        raise HTTPException(status_code=400,
                            detail="Apologies data is not available for {} at {}".format(lake, time))
    with netCDF4.Dataset(file) as nc:
        converted_time = functions.convert_to_unit(origin, nc.variables["time"].units)
        time_index = functions.get_closest_index(converted_time, np.array(nc.variables["time"][:]))
        depth_index = functions.get_closest_index(depth, np.array(nc.variables["depth"][:]))
        time = nc.variables["time"][time_index].tolist()
        depth = nc.variables["depth"][depth_index].tolist()
        lat_grid, lng_grid = nc.variables["lat"][:], nc.variables["lng"][:]
        output = {"time": functions.alplakes_time(time, nc.variables["time"].units),
                  "depth": {"description": "Distance from the surface to the closest grid point to requested depth",
                            "units": nc.variables["depth"].units,
                            "data": depth},
                  "lat": functions.filter_variable(lat_grid, decimals=5, nodata=np.nan),
                  "lng": functions.filter_variable(lng_grid, decimals=5, nodata=np.nan),
                  "variables": {}
                  }
        if "temperature" in variables:
            output["variables"]["temperature"] = {"data": functions.filter_variable(nc.variables["t"][time_index, depth_index, :]), "unit": "degC",
                                                  "description": "Water temperature"}
        if "velocity" in variables:
            output["variables"]["u"] = {"data": functions.filter_variable(nc.variables["u"][time_index, depth_index, :], decimals=5), "unit": "m/s",
                                        "description": "Eastward flow velocity"}
            output["variables"]["v"] = {"data": functions.filter_variable(nc.variables["v"][time_index, depth_index, :], decimals=5), "unit": "m/s",
                                        "description": "Northward flow velocity"}

    return output


def get_simulations_layer_alplakes(filesystem, model, lake, variable, start, end, depth):
    if model == "delft3d-flow":
        return get_simulations_layer_alplakes_delft3dflow(filesystem, lake, variable, start, end, depth)
    elif model == "mitgcm":
        return get_simulations_layer_alplakes_mitgcm(filesystem, lake, variable, start, end, depth)
    else:
        raise HTTPException(status_code=400,
                            detail="Apologies data is not available for {}".format(model))


def get_simulations_layer_alplakes_delft3dflow(filesystem, lake, variable, start, end, depth):
    model = "delft3d-flow"
    lakes = os.path.join(filesystem, "media/simulations", model, "results")
    if not os.path.isdir(os.path.join(lakes, lake)):
        raise HTTPException(status_code=400,
                            detail="{} simulation results are not available for {} please select from: [{}]"
                            .format(model, lake, ", ".join(os.listdir(lakes))))
    weeks = functions.sundays_between_dates(datetime.strptime(start[0:8], "%Y%m%d").replace(tzinfo=timezone.utc),
                                            datetime.strptime(end[0:8], "%Y%m%d").replace(tzinfo=timezone.utc))

    for week in weeks:
        if not os.path.isfile(os.path.join(lakes, lake, "{}.nc".format(week.strftime("%Y%m%d")))):
            raise HTTPException(status_code=400,
                                detail="Apologies data is not available for {} week starting {}".format(lake, week))

    start_datetime = datetime.strptime(start[0:10], "%Y%m%d%H").replace(tzinfo=timezone.utc)
    end_datetime = datetime.strptime(end[0:10], "%Y%m%d%H").replace(tzinfo=timezone.utc)
    out = None
    times = None
    for week in weeks:
        with netCDF4.Dataset(os.path.join(lakes, lake, "{}.nc".format(week.strftime("%Y%m%d")))) as nc:
            if variable == "geometry":
                lat_grid, lng_grid = functions.coordinates_to_latlng(nc.variables["XZ"][:], nc.variables["YZ"][:])
                geometry = np.concatenate((lat_grid, lng_grid), axis=1)
                return '\n'.join(','.join('%0.8f' % x for x in y) for y in geometry).replace("nan", "")
            time = np.array(nc.variables["time"][:])
            min_time = np.min(time)
            max_time = np.max(time)
            start_time = functions.convert_to_unit(start_datetime, nc.variables["time"].units)
            end_time = functions.convert_to_unit(end_datetime, nc.variables["time"].units)
            if start_time > max_time:
                continue
            if min_time <= start_time:
                time_index_start = functions.get_closest_index(start_time, time)
            else:
                time_index_start = 0
            if min_time <= end_time <= max_time:
                time_index_end = functions.get_closest_index(end_time, time) + 1
            else:
                time_index_end = len(time)

            depth_index = functions.get_closest_index(depth, np.array(nc.variables["ZK_LYR"][:]) * -1)

            if variable == "temperature":
                f = '%0.2f'
                p = functions.alplakes_variable(
                    nc.variables["R1"][time_index_start:time_index_end, 0, depth_index, :])
            elif variable == "velocity":
                f = '%0.5f'
                p = functions.alplakes_velocity(
                    nc.variables["U1"][time_index_start:time_index_end, depth_index, :],
                    nc.variables["V1"][time_index_start:time_index_end, depth_index, :],
                    nc.variables["ALFAS"][:])
            elif variable == "thermocline":
                if "THERMOCLINE" in nc.variables.keys():
                    f = '%0.2f'
                    p = functions.alplakes_variable(
                        nc.variables["THERMOCLINE"][time_index_start:time_index_end, :])
                else:
                    raise HTTPException(status_code=400,
                                        detail="Thermocline not available for this dataset. Please try another variable.")
            else:
                raise HTTPException(status_code=400,
                                    detail="Variable {} not recognised, please select from: [geometry, temperature, "
                                           "velocity, thermocline]".format(variable))
            t = np.array([functions.convert_from_unit(x, nc.variables["time"].units).strftime("%Y%m%d%H%M") for x in time[time_index_start:time_index_end]])
            if out is None:
                out = p
                times = t
            else:
                out = np.concatenate((out, p), axis=0)
                times = np.concatenate((times, t), axis=0)

    shape = out.shape
    string_arr = ""
    for timestep in range(shape[0]):
        string_arr += (times[timestep] + "\n" + '\n'.join(','.join(f % x for x in y) for y in out[timestep, :]).replace(
            "nan", "") + "\n")

    return string_arr


def get_simulations_layer_alplakes_mitgcm(filesystem, lake, variable, start, end, depth):
    model = "mitgcm"
    lakes = os.path.join(filesystem, "media/simulations", model, "results")
    if not os.path.isdir(os.path.join(lakes, lake)):
        raise HTTPException(status_code=400,
                            detail="{} simulation results are not available for {} please select from: [{}]"
                            .format(model, lake, ", ".join(os.listdir(lakes))))
    weeks = functions.sundays_between_dates(datetime.strptime(start[0:8], "%Y%m%d").replace(tzinfo=timezone.utc),
                                            datetime.strptime(end[0:8], "%Y%m%d").replace(tzinfo=timezone.utc))

    for week in weeks:
        if not os.path.isfile(os.path.join(lakes, lake, "{}.nc".format(week.strftime("%Y%m%d")))):
            raise HTTPException(status_code=400,
                                detail="Apologies data is not available for {} week starting {}".format(lake, week))

    start_datetime = datetime.strptime(start[0:10], "%Y%m%d%H").replace(tzinfo=timezone.utc)
    end_datetime = datetime.strptime(end[0:10], "%Y%m%d%H").replace(tzinfo=timezone.utc)
    out = None
    times = None
    for week in weeks:
        with netCDF4.Dataset(os.path.join(lakes, lake, "{}.nc".format(week.strftime("%Y%m%d")))) as nc:
            if variable == "geometry":
                geometry = np.concatenate((nc.variables["lat"][:], nc.variables["lng"][:]), axis=1)
                return '\n'.join(','.join('%0.8f' % x for x in y) for y in geometry).replace("nan", "")
            time = np.array(nc.variables["time"][:])
            min_time = np.min(time)
            max_time = np.max(time)
            start_time = functions.convert_to_unit(start_datetime, nc.variables["time"].units)
            end_time = functions.convert_to_unit(end_datetime, nc.variables["time"].units)
            if start_time > max_time:
                continue
            if min_time <= start_time:
                time_index_start = functions.get_closest_index(start_time, time)
            else:
                time_index_start = 0
            if min_time <= end_time <= max_time:
                time_index_end = functions.get_closest_index(end_time, time) + 1
            else:
                time_index_end = len(time)

            depth_index = functions.get_closest_index(depth, np.array(nc.variables["depth"][:]))

            if variable == "temperature":
                f = '%0.2f'
                p = functions.alplakes_variable(
                    nc.variables["t"][time_index_start:time_index_end, depth_index, :])
            elif variable == "velocity":
                f = '%0.5f'
                p = functions.alplakes_variable(np.concatenate((nc.variables["u"][time_index_start:time_index_end, depth_index, :],
                                    nc.variables["v"][time_index_start:time_index_end, depth_index, :]), axis=2))
            elif variable == "thermocline":
                if "thermocline" in nc.variables.keys():
                    f = '%0.2f'
                    p = functions.alplakes_variable(
                        nc.variables["thermocline"][time_index_start:time_index_end, :])
                else:
                    raise HTTPException(status_code=400,
                                        detail="Thermocline not available for this dataset. Please try another variable.")
            else:
                raise HTTPException(status_code=400,
                                    detail="Variable {} not recognised, please select from: [geometry, temperature, "
                                           "velocity, thermocline]".format(variable))
            t = np.array([functions.convert_from_unit(x, nc.variables["time"].units).strftime("%Y%m%d%H%M") for x in time[time_index_start:time_index_end]])
            if out is None:
                out = p
                times = t
            else:
                out = np.concatenate((out, p), axis=0)
                times = np.concatenate((times, t), axis=0)

    shape = out.shape
    string_arr = ""
    for timestep in range(shape[0]):
        string_arr += (times[timestep] + "\n" + '\n'.join(','.join(f % x for x in y) for y in out[timestep, :]).replace(
            "nan", "") + "\n")

    return string_arr


def get_simulations_layer_average_temperature(filesystem, model, lake, start, end, depth):
    if model == "delft3d-flow":
        return get_simulations_layer_average_temperature_delft3dflow(filesystem, lake, start, end, depth)
    elif model == "mitgcm":
        return get_simulations_layer_average_temperature_mitgcm(filesystem, lake, start, end, depth)
    else:
        raise HTTPException(status_code=400,
                            detail="Apologies data is not available for {}".format(model))


def get_simulations_layer_average_temperature_delft3dflow(filesystem, lake, start, end, depth, nodata=-999.0):
    model = "delft3d-flow"
    lakes = os.path.join(filesystem, "media/simulations", model, "results")
    if not os.path.isdir(os.path.join(lakes, lake)):
        raise HTTPException(status_code=400,
                            detail="{} simulation results are not available for {} please select from: [{}]"
                            .format(model, lake, ", ".join(os.listdir(lakes))))
    weeks = functions.sundays_between_dates(datetime.strptime(start[0:8], "%Y%m%d").replace(tzinfo=timezone.utc),
                                            datetime.strptime(end[0:8], "%Y%m%d").replace(tzinfo=timezone.utc))
    files = [os.path.join(lakes, lake, "{}.nc".format(week.strftime("%Y%m%d"))) for week in weeks]

    files = [file for file in files if os.path.isfile(file)]
    if len(files) == 0:
        raise HTTPException(status_code=400, detail="Apologies data is not available for your requested period")

    start_datetime = datetime.strptime(start[0:10], "%Y%m%d%H").replace(tzinfo=timezone.utc)
    end_datetime = datetime.strptime(end[0:10], "%Y%m%d%H").replace(tzinfo=timezone.utc)

    with xr.open_mfdataset(files, data_vars='minimal', compat='override', coords='minimal') as ds:
        ds['time'] = ds.indexes['time'].tz_localize('UTC')
        ds = ds.sel(time=slice(start_datetime, end_datetime))
        if len(ds['time']) == 0:
            raise HTTPException(status_code=400,
                                detail="No timesteps available between {} and {}".format(start, end))
        z = ds.ZK_LYR[0, :].values * -1 if len(ds.ZK_LYR.shape) == 2 else ds.ZK_LYR[:].values * -1
        depth_index = functions.get_closest_index(depth, z)
        depth = float(z[depth_index])
        time = functions.alplakes_time(ds.time.values, "nano")
        t_arr = ds.R1.isel(KMAXOUT_RESTR=depth_index, LSTSCI=0)
        t_arr = t_arr.where(t_arr != nodata, np.nan)
        t = t_arr.mean(dim=['M', 'N'], skipna=True).values
        output = {"time": time,
                  "depth": {"data": depth, "unit": "m",
                            "description": "Distance from the surface to the closest grid point to requested depth"},
                  "variable": {"data": functions.filter_variable(t), "unit": "degC", "description": "Water temperature"}
                  }
    return output


def get_simulations_layer_average_temperature_mitgcm(filesystem, lake, start, end, depth, nodata=-999.0):
    model = "mitgcm"
    lakes = os.path.join(filesystem, "media/simulations", model, "results")
    if not os.path.isdir(os.path.join(lakes, lake)):
        raise HTTPException(status_code=400,
                            detail="{} simulation results are not available for {} please select from: [{}]"
                            .format(model, lake, ", ".join(os.listdir(lakes))))
    weeks = functions.sundays_between_dates(datetime.strptime(start[0:8], "%Y%m%d").replace(tzinfo=timezone.utc),
                                            datetime.strptime(end[0:8], "%Y%m%d").replace(tzinfo=timezone.utc))
    files = [os.path.join(lakes, lake, "{}.nc".format(week.strftime("%Y%m%d"))) for week in weeks]

    files = [file for file in files if os.path.isfile(file)]
    if len(files) == 0:
        raise HTTPException(status_code=400, detail="Apologies data is not available for your requested period")

    start_datetime = datetime.strptime(start[0:10], "%Y%m%d%H").replace(tzinfo=timezone.utc)
    end_datetime = datetime.strptime(end[0:10], "%Y%m%d%H").replace(tzinfo=timezone.utc)

    with xr.open_mfdataset(files, data_vars='minimal', compat='override', coords='minimal') as ds:
        ds['time'] = ds.indexes['time'].tz_localize('UTC')
        ds = ds.sel(time=slice(start_datetime, end_datetime))
        if len(ds['time']) == 0:
            raise HTTPException(status_code=400,
                                detail="No timesteps available between {} and {}".format(start, end))
        z = ds.depth[0, :].values if len(ds.depth.shape) == 2 else ds.depth[:].values
        depth_index = functions.get_closest_index(depth, z)
        depth = float(z[depth_index])
        time = functions.alplakes_time(ds.time.values, "nano")
        t_arr = ds.t.isel(depth=depth_index)
        t_arr = t_arr.where(t_arr != nodata, np.nan)
        t = t_arr.mean(dim=['X', 'Y'], skipna=True).values
        output = {"time": time,
                  "depth": {"data": depth, "unit": "m",
                            "description": "Distance from the surface to the closest grid point to requested depth"},
                  "variable": {"data": functions.filter_variable(t), "unit": "degC", "description": "Water temperature"}
                  }
    return output


def get_simulations_average_bottom_temperature(filesystem, model, lake, start, end):
    if model == "delft3d-flow":
        return get_simulations_average_bottom_temperature_delft3dflow(filesystem, lake, start, end)
    elif model == "mitgcm":
        return get_simulations_average_bottom_temperature_mitgcm(filesystem, lake, start, end)
    else:
        raise HTTPException(status_code=400,
                            detail="Apologies data is not available for {}".format(model))


def get_simulations_average_bottom_temperature_delft3dflow(filesystem, lake, start, end, nodata=-999.0):
    model = "delft3d-flow"
    lakes = os.path.join(filesystem, "media/simulations", model, "results")
    if not os.path.isdir(os.path.join(lakes, lake)):
        raise HTTPException(status_code=400,
                            detail="{} simulation results are not available for {} please select from: [{}]"
                            .format(model, lake, ", ".join(os.listdir(lakes))))
    weeks = functions.sundays_between_dates(datetime.strptime(start[0:8], "%Y%m%d").replace(tzinfo=timezone.utc),
                                            datetime.strptime(end[0:8], "%Y%m%d").replace(tzinfo=timezone.utc))
    files = [os.path.join(lakes, lake, "{}.nc".format(week.strftime("%Y%m%d"))) for week in weeks]

    files = [file for file in files if os.path.isfile(file)]
    if len(files) == 0:
        raise HTTPException(status_code=400, detail="Apologies data is not available for your requested period")

    start_datetime = datetime.strptime(start[0:10], "%Y%m%d%H").replace(tzinfo=timezone.utc)
    end_datetime = datetime.strptime(end[0:10], "%Y%m%d%H").replace(tzinfo=timezone.utc)

    with xr.open_mfdataset(files, data_vars='minimal', compat='override', coords='minimal') as ds:
        ds['time'] = ds.indexes['time'].tz_localize('UTC')
        ds = ds.sel(time=slice(start_datetime, end_datetime))
        if len(ds['time']) == 0:
            raise HTTPException(status_code=400,
                                detail="No timesteps available between {} and {}".format(start, end))

        t = ds["R1"].values[:, 0, :]
        t[t == nodata] = np.nan
        valid_mask = ~np.isnan(t[0])
        bottom_indices = np.argmax(valid_mask, axis=0)
        no_valid_data_mask = ds["KCS"].values == 0
        if len(no_valid_data_mask.shape) > 2:
            no_valid_data_mask = no_valid_data_mask[0]
        bottom_indices[no_valid_data_mask] = -1
        rows, cols = np.meshgrid(np.arange(t.shape[2]), np.arange(t.shape[3]), indexing='ij')
        result = functions.safe_nanmean(t[:, bottom_indices, rows, cols], axis=0)

        lat_grid, lng_grid = functions.coordinates_to_latlng(ds["XZ"].values, ds["YZ"].values)

        output = {"variable": {"data": functions.filter_variable(result), "unit": "degC", "description": "Average bottom temperature"},
                  "lat": functions.filter_variable(lat_grid, decimals=5, nodata=np.nan),
                  "lng": functions.filter_variable(lng_grid, decimals=5, nodata=np.nan),
                  }
    return output


def get_simulations_average_bottom_temperature_mitgcm(filesystem, lake, start, end, nodata=-999.0):
    model = "mitgcm"
    lakes = os.path.join(filesystem, "media/simulations", model, "results")
    if not os.path.isdir(os.path.join(lakes, lake)):
        raise HTTPException(status_code=400,
                            detail="{} simulation results are not available for {} please select from: [{}]"
                            .format(model, lake, ", ".join(os.listdir(lakes))))
    weeks = functions.sundays_between_dates(datetime.strptime(start[0:8], "%Y%m%d").replace(tzinfo=timezone.utc),
                                            datetime.strptime(end[0:8], "%Y%m%d").replace(tzinfo=timezone.utc))
    files = [os.path.join(lakes, lake, "{}.nc".format(week.strftime("%Y%m%d"))) for week in weeks]

    files = [file for file in files if os.path.isfile(file)]
    if len(files) == 0:
        raise HTTPException(status_code=400, detail="Apologies data is not available for your requested period")

    start_datetime = datetime.strptime(start[0:10], "%Y%m%d%H").replace(tzinfo=timezone.utc)
    end_datetime = datetime.strptime(end[0:10], "%Y%m%d%H").replace(tzinfo=timezone.utc)

    with xr.open_mfdataset(files, data_vars='minimal', compat='override', coords='minimal') as ds:
        ds['time'] = ds.indexes['time'].tz_localize('UTC')
        ds = ds.sel(time=slice(start_datetime, end_datetime))
        if len(ds['time']) == 0:
            raise HTTPException(status_code=400,
                                detail="No timesteps available between {} and {}".format(start, end))

        t = ds["t"].values[:]
        t[t == nodata] = np.nan
        valid_mask = ~np.isnan(t[0])
        reversed_valid_mask = valid_mask[::-1, :, :]
        deepest_valid_indices_reversed = np.argmax(reversed_valid_mask, axis=0)
        depth_size = t.shape[1]
        deepest_valid_indices = depth_size - 1 - deepest_valid_indices_reversed
        all_nan_mask = ~np.any(valid_mask, axis=0)
        deepest_valid_indices[all_nan_mask] = -1
        rows, cols = np.meshgrid(np.arange(t.shape[2]), np.arange(t.shape[3]), indexing='ij')
        result = functions.safe_nanmean(t[:, deepest_valid_indices, rows, cols], axis=0)
        lat_grid, lng_grid = ds["lat"].values, ds["lng"].values

        output = {"variable": {"data": functions.filter_variable(result), "unit": "degC", "description": "Average bottom temperature"},
                  "lat": functions.filter_variable(lat_grid, decimals=5, nodata=np.nan),
                  "lng": functions.filter_variable(lng_grid, decimals=5, nodata=np.nan),
                  }
    return output


def get_simulations_profile(filesystem, model, lake, dt, latitude, longitude, variables):
    if model == "delft3d-flow":
        return get_simulations_profile_delft3dflow(filesystem, lake, dt, latitude, longitude, variables)
    elif model == "mitgcm":
        return get_simulations_profile_mitgcm(filesystem, lake, dt, latitude, longitude, variables)
    else:
        raise HTTPException(status_code=400,
                            detail="Apologies profile extraction not available for {}".format(model))


def get_simulations_profile_delft3dflow(filesystem, lake, dt, latitude, longitude, variables):
    model = "delft3d-flow"
    lakes = os.path.join(filesystem, "media/simulations", model, "results")
    variables = [v.lower() for v in variables]
    if not os.path.isdir(os.path.join(lakes, lake)):
        raise HTTPException(status_code=400,
                            detail="{} simulation results are not available for {} please select from: [{}]"
                            .format(model, lake, ", ".join(os.listdir(lakes))))
    origin = datetime.strptime(dt, "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
    last_sunday = origin + relativedelta(weekday=SU(-1))
    if os.path.isfile(os.path.join(lakes, lake, "{}.nc".format(last_sunday.strftime("%Y%m%d")))):
        file = os.path.join(lakes, lake, "{}.nc".format(last_sunday.strftime("%Y%m%d")))
    else:
        raise HTTPException(status_code=400,
                            detail="Apologies data is not available for {} at {}".format(lake, dt))
    with netCDF4.Dataset(file) as nc:
        converted_time = functions.convert_to_unit(origin, nc.variables["time"].units)
        time = np.array(nc.variables["time"][:])
        if converted_time > np.nanmax(time):
            raise HTTPException(status_code=400,
                                detail="Apologies data is not available for {} at {}".format(lake, dt))
        time_index = functions.get_closest_index(converted_time, time)

        depth = (np.array(nc.variables["ZK_LYR"][:]) * -1).tolist()
        lat_grid, lng_grid = functions.coordinates_to_latlng(nc.variables["XZ"][:], nc.variables["YZ"][:])
        x_index, y_index, distance = functions.get_closest_location(latitude, longitude, lat_grid, lng_grid)

        t = functions.filter_variable(nc.variables["R1"][time_index, 0, :, x_index, y_index])
        u, v, = functions.rotate_velocity(nc.variables["U1"][time_index, :, x_index, y_index],
                                          nc.variables["V1"][time_index, :, x_index, y_index],
                                          nc.variables["ALFAS"][x_index, y_index])

        index = 0
        for i in range(len(t)):
            if not t[i] is None:
                index = i
                break

        depth = depth[index:]
        t = t[index:]
        u = u[index:]
        v = v[index:]

        output = {"time": functions.alplakes_time(time[time_index], nc.variables["time"].units),
                  "lat": lat_grid[x_index, y_index],
                  "lng": lng_grid[x_index, y_index],
                  "distance": {"data": distance, "unit": "m", "description": "Distance from requested location to center of closest grid point"},
                  "depth": {"data": functions.filter_variable(depth), "unit": "m", "description": "Distance from the surface to the closest grid point to requested depth"},
                  "variables": {}}
        if "temperature" in variables:
            output["variables"]["temperature"] = {"data": t, "unit": "degC", "description": "Water temperature"}
        if "velocity" in variables:
            output["variables"]["u"] = {"data": functions.filter_variable(u, decimals=5), "unit": nc.variables["U1"].units,
                  "description": "Eastward flow velocity"}
            output["variables"]["v"] = {"data": functions.filter_variable(v, decimals=5), "unit": nc.variables["V1"].units,
                  "description": "Northward flow velocity"}
    return output


def get_simulations_profile_mitgcm(filesystem, lake, dt, latitude, longitude, variables):
    model = "mitgcm"
    lakes = os.path.join(filesystem, "media/simulations", model, "results")
    variables = [v.lower() for v in variables]
    if not os.path.isdir(os.path.join(lakes, lake)):
        raise HTTPException(status_code=400,
                            detail="{} simulation results are not available for {} please select from: [{}]"
                            .format(model, lake, ", ".join(os.listdir(lakes))))
    origin = datetime.strptime(dt, "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
    last_sunday = origin + relativedelta(weekday=SU(-1))
    if os.path.isfile(os.path.join(lakes, lake, "{}.nc".format(last_sunday.strftime("%Y%m%d")))):
        file = os.path.join(lakes, lake, "{}.nc".format(last_sunday.strftime("%Y%m%d")))
    else:
        raise HTTPException(status_code=400,
                            detail="Apologies data is not available for {} at {}".format(lake, dt))
    with netCDF4.Dataset(file) as nc:
        converted_time = functions.convert_to_unit(origin, nc.variables["time"].units)
        time = np.array(nc.variables["time"][:])
        if converted_time > np.nanmax(time):
            raise HTTPException(status_code=400,
                                detail="Apologies data is not available for {} at {}".format(lake, dt))
        time_index = functions.get_closest_index(converted_time, time)

        depth = (np.array(nc.variables["depth"][:])).tolist()
        lat_grid, lng_grid = nc.variables["lat"][:], nc.variables["lng"][:]
        mask = np.isnan(nc.variables["t"][0,0,:])
        lat_grid[mask] = np.nan
        lng_grid[mask] = np.nan
        x_index, y_index, distance = functions.get_closest_location(latitude, longitude, lat_grid, lng_grid, yx=True)

        t = functions.filter_variable(nc.variables["t"][time_index, :, y_index, x_index])
        u = functions.filter_variable(nc.variables["u"][time_index, :, y_index, x_index])
        v = functions.filter_variable(nc.variables["v"][time_index, :, y_index, x_index])

        index = 0
        for i in range(len(t)):
            if not t[i] is None:
                index = i
                break

        depth = depth[index:]
        t = t[index:]
        u = u[index:]
        v = v[index:]

        output = {"time": functions.alplakes_time(time[time_index], nc.variables["time"].units),
                  "lat": lat_grid[y_index, x_index],
                  "lng": lng_grid[y_index, x_index],
                  "distance": {"data": distance, "unit": "m", "description": "Distance from requested location to center of closest grid point"},
                  "depth": {"data": functions.filter_variable(depth), "unit": "m", "description": "Distance from the surface to the closest grid point to requested depth"},
                  "variables": {}}
        if "temperature" in variables:
            output["variables"]["temperature"] = {"data": t, "unit": "degC", "description": "Water temperature"}
        if "velocity" in variables:
            output["variables"]["u"] = {"data": functions.filter_variable(u, decimals=5), "unit": nc.variables["u"].units,
                  "description": "Eastward flow velocity"}
            output["variables"]["v"] = {"data": functions.filter_variable(v, decimals=5), "unit": nc.variables["v"].units,
                  "description": "Northward flow velocity"}
    return output


def get_simulations_depthtime(filesystem, model, lake, start, end, latitude, longitude, variables):
    if model == "delft3d-flow":
        return get_simulations_depthtime_delft3dflow(filesystem, lake, start, end, latitude, longitude, variables)
    if model == "mitgcm":
        return get_simulations_depthtime_mitgcm(filesystem, lake, start, end, latitude, longitude, variables)
    else:
        raise HTTPException(status_code=400,
                            detail="Apologies profile extraction not available for {}".format(model))


def get_simulations_depthtime_delft3dflow(filesystem, lake, start, end, latitude, longitude, variables, nodata=-999.0):
    model = "delft3d-flow"
    lakes = os.path.join(filesystem, "media/simulations", model, "results")
    variables = [v.lower() for v in variables]
    if not os.path.isdir(os.path.join(lakes, lake)):
        raise HTTPException(status_code=400,
                            detail="{} simulation results are not available for {} please select from: [{}]"
                            .format(model, lake, ", ".join(os.listdir(lakes))))
    weeks = functions.sundays_between_dates(datetime.strptime(start[0:8], "%Y%m%d").replace(tzinfo=timezone.utc),
                                            datetime.strptime(end[0:8], "%Y%m%d").replace(tzinfo=timezone.utc))
    files = [os.path.join(lakes, lake, "{}.nc".format(week.strftime("%Y%m%d"))) for week in weeks]

    files = [file for file in files if os.path.isfile(file)]
    if len(files) == 0:
        raise HTTPException(status_code=400, detail="Apologies data is not available for your requested period")

    start_datetime = datetime.strptime(start, "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
    end_datetime = datetime.strptime(end, "%Y%m%d%H%M").replace(tzinfo=timezone.utc)

    with xr.open_mfdataset(files, data_vars='minimal', compat='override', coords='minimal') as ds:
        ds['time'] = ds.indexes['time'].tz_localize('UTC')
        ds = ds.sel(time=slice(start_datetime, end_datetime))
        if len(ds['time']) == 0:
            raise HTTPException(status_code=400,
                                detail="No timesteps available between {} and {}".format(start, end))
        lat_grid, lng_grid = functions.coordinates_to_latlng(ds.XZ[:].values, ds.YZ[:].values)
        x_index, y_index, distance = functions.get_closest_location(latitude, longitude, lat_grid, lng_grid)
        t = ds.R1.isel(time=0, M=x_index, N=y_index, LSTSCI=0).values
        depth = ds.ZK_LYR[0, :].values * -1 if len(ds.ZK_LYR.shape) == 2 else ds.ZK_LYR[:].values * -1
        valid_depths = t != nodata
        depth = depth[valid_depths]
        ds = ds.sel(KMAXOUT_RESTR=valid_depths)
        time = functions.alplakes_time(ds.time.values, "nano")
        output = {"time": time,
                  "lat": lat_grid[x_index, y_index],
                  "lng": lng_grid[x_index, y_index],
                  "depth": {"data": functions.filter_variable(depth), "unit": "m",
                            "description": "Distance from the surface"},
                  "distance": {"data": distance, "unit": "m",
                               "description": "Distance from requested location to center of closest grid point"},
                  "variables": {}
                  }
        if "temperature" in variables:
            t = ds.R1.isel(M=x_index, N=y_index, LSTSCI=0).transpose('KMAXOUT_RESTR', 'time').values
            output["variables"]["temperature"] = {"data": functions.filter_variable(t), "unit": "degC",
                                                  "description": "Water temperature"}
        if "velocity" in variables:
            if 'time' in ds.ALFAS.coords:
                alfas = ds.ALFAS.isel(M=x_index, N=y_index).values[np.newaxis, :]
            else:
                alfas = np.full(len(time), ds.ALFAS.isel(M=x_index, N=y_index).values)
            u, v, = functions.rotate_velocity(
                ds.U1.isel(MC=x_index, N=y_index).transpose('KMAXOUT_RESTR', 'time').values,
                ds.V1.isel(M=x_index, NC=y_index).transpose('KMAXOUT_RESTR', 'time').values,
                alfas)
            output["variables"]["u"] = {"data": functions.filter_variable(u, decimals=5), "unit": "m/s",
                                        "description": "Eastward flow velocity"}
            output["variables"]["v"] = {"data": functions.filter_variable(v, decimals=5), "unit": "m/s",
                                        "description": "Northward flow velocity"}
    return output


def get_simulations_depthtime_mitgcm(filesystem, lake, start, end, latitude, longitude, variables, nodata=-999.0):
    model = "mitgcm"
    lakes = os.path.join(filesystem, "media/simulations", model, "results")
    variables = [v.lower() for v in variables]
    if not os.path.isdir(os.path.join(lakes, lake)):
        raise HTTPException(status_code=400,
                            detail="{} simulation results are not available for {} please select from: [{}]"
                            .format(model, lake, ", ".join(os.listdir(lakes))))
    weeks = functions.sundays_between_dates(datetime.strptime(start[0:8], "%Y%m%d").replace(tzinfo=timezone.utc),
                                            datetime.strptime(end[0:8], "%Y%m%d").replace(tzinfo=timezone.utc))
    files = [os.path.join(lakes, lake, "{}.nc".format(week.strftime("%Y%m%d"))) for week in weeks]

    files = [file for file in files if os.path.isfile(file)]
    if len(files) == 0:
        raise HTTPException(status_code=400, detail="Apologies data is not available for your requested period")

    start_datetime = datetime.strptime(start, "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
    end_datetime = datetime.strptime(end, "%Y%m%d%H%M").replace(tzinfo=timezone.utc)

    with xr.open_mfdataset(files, data_vars='minimal', compat='override', coords='minimal') as ds:
        ds['time'] = ds.indexes['time'].tz_localize('UTC')
        ds = ds.sel(time=slice(start_datetime, end_datetime))
        if len(ds['time']) == 0:
            raise HTTPException(status_code=400,
                                detail="No timesteps available between {} and {}".format(start, end))
        if len(ds.lat.shape) == 2:
            lat_grid, lng_grid = ds.lat.values, ds.lng.values
        else:
            lat_grid, lng_grid = ds.lat.isel(time=0).values, ds.lng.isel(time=0).values
        mask = np.isnan(ds.t.isel(time=0, depth=0).values)
        lat_grid[mask] = np.nan
        lng_grid[mask] = np.nan
        x_index, y_index, distance = functions.get_closest_location(latitude, longitude, lat_grid, lng_grid, yx=True)
        t = ds.t.isel(time=0, X=x_index, Y=y_index).values
        depth = ds.depth[0, :].values if len(ds.depth.shape) == 2 else ds.depth[:].values
        valid_depths = ~np.isnan(t)
        depth = depth[valid_depths]
        ds = ds.sel(depth=valid_depths)
        time = functions.alplakes_time(ds.time.values, "nano")
        output = {"time": time,
                  "lat": lat_grid[y_index, x_index],
                  "lng": lng_grid[y_index, x_index],
                  "depth": {"data": functions.filter_variable(depth), "unit": "m",
                            "description": "Distance from the surface"},
                  "distance": {"data": distance, "unit": "m",
                               "description": "Distance from requested location to center of closest grid point"},
                  "variables": {}
                  }
        if "temperature" in variables:
            t = ds.t.isel(X=x_index, Y=y_index).transpose('depth', 'time').values
            output["variables"]["temperature"] = {"data": functions.filter_variable(t), "unit": "degC",
                                                  "description": "Water temperature"}
        if "velocity" in variables:
            u = ds.u.isel(X=x_index, Y=y_index).transpose('depth', 'time').values
            v = ds.v.isel(X=x_index, Y=y_index).transpose('depth', 'time').values
            output["variables"]["u"] = {"data": functions.filter_variable(u, decimals=5), "unit": "m/s",
                                        "description": "Eastward flow velocity"}
            output["variables"]["v"] = {"data": functions.filter_variable(v, decimals=5), "unit": "m/s",
                                        "description": "Northward flow velocity"}
    return output


def get_simulations_transect(filesystem, model, lake, dt, latitude_list, longitude_list, variables):
    if model == "delft3d-flow":
        return get_simulations_transect_delft3dflow(filesystem, lake, dt, latitude_list, longitude_list, variables)
    if model == "mitgcm":
        return get_simulations_transect_mitgcm(filesystem, lake, dt, latitude_list, longitude_list, variables)
    else:
        raise HTTPException(status_code=400,
                            detail="Apologies profile extraction not available for {}".format(model))


def get_simulations_transect_delft3dflow(filesystem, lake, time, latitude_str, longitude_str, variables, nodata=-999.0):
    model = "delft3d-flow"
    variables = [v.lower() for v in variables]
    latitude_list = [float(x) for x in latitude_str.replace(" ", "").split(",")]
    longitude_list = [float(x) for x in longitude_str.replace(" ", "").split(",")]

    if len(latitude_list) < 2 or len(latitude_list) != len(longitude_list):
        raise HTTPException(status_code=400,
                            detail="At least two valid points should be provided.")

    origin = datetime.strptime(time, "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
    last_sunday = origin + relativedelta(weekday=SU(-1))
    previous_sunday = last_sunday - timedelta(days=7)
    lakes = os.path.join(filesystem, "media/simulations", model, "results")
    if not os.path.isdir(os.path.join(lakes, lake)):
        raise HTTPException(status_code=400,
                            detail="{} simulation results are not available for {} please select from: [{}]"
                            .format(model, lake, ", ".join(os.listdir(lakes))))
    if os.path.isfile(os.path.join(lakes, lake, "{}.nc".format(last_sunday.strftime("%Y%m%d")))):
        file = os.path.join(lakes, lake, "{}.nc".format(last_sunday.strftime("%Y%m%d")))
    elif os.path.isfile(os.path.join(lakes, lake, "{}.nc".format(previous_sunday.strftime("%Y%m%d")))):
        file = os.path.join(lakes, lake, "{}.nc".format(previous_sunday.strftime("%Y%m%d")))
    else:
        raise HTTPException(status_code=400,
                            detail="Apologies data is not available for {} at {}".format(lake, time))

    with xr.open_mfdataset(file) as ds:
        ds['time'] = ds.indexes['time'].tz_localize('UTC')
        time_index = functions.get_closest_index(functions.convert_to_unit(origin, "nano"), ds["time"].values)
        ds = ds.isel(time=time_index)
        x = ds.XZ[:].values
        y = ds.YZ[:].values
        if len(ds.ZK_LYR.shape) == 2:
            z = ds.ZK_LYR[0, :].values * - 1
        else:
            z = ds.ZK_LYR[:].values * - 1
        grid_spacing = functions.center_grid_spacing(x, y)
        projection = functions.identify_projection(np.max(x), np.max(y))
        if projection == "WGS84":
            raise HTTPException(status_code=400, detail="Method not implemented for models with projection WGS84")
        x_list, y_list = functions.latlng_to_projection(latitude_list, longitude_list, projection)
        indexes = np.where((x >= np.min(x_list) - 2 * grid_spacing) &
                           (x <= np.max(x_list) + 2 * grid_spacing) &
                           (y >= np.min(y_list) - 2 * grid_spacing) &
                           (y <= np.max(y_list) + 2 * grid_spacing))
        start = 0
        xi_arr, yi_arr, sp_arr, vd_arr = np.array([]), np.array([]), np.array([]), np.array([])
        for i in range(len(x_list) - 1):
            xi, yi, sp, vd, distance = functions.line_segments(x_list[i], y_list[i], x_list[i + 1], y_list[i + 1], x, y,
                                                               indexes, start, grid_spacing)
            start = start + distance
            xi_arr = np.concatenate((xi_arr, xi), axis=0)
            yi_arr = np.concatenate((yi_arr, yi), axis=0)
            sp_arr = np.concatenate((sp_arr, sp), axis=0)
            vd_arr = np.concatenate((vd_arr, vd), axis=0)

        xi_arr = xi_arr.astype(int)
        yi_arr = yi_arr.astype(int)
        lat_arr, lng_arr = functions.projection_to_latlng(x[xi_arr, yi_arr], y[xi_arr, yi_arr], projection)

        xi = xr.DataArray(xi_arr)
        yi = xr.DataArray(yi_arr)

        t = ds.R1.isel(M=xi, N=yi, LSTSCI=0).values
        valid_depths = ~np.all(t == nodata, axis=1)
        depth = z[valid_depths]
        ds = ds.sel(KMAXOUT_RESTR=valid_depths)
        time_value = ds.time.values
        if isinstance(time_value, np.datetime64):
            time_value = pd.Timestamp(time_value).tz_localize('UTC')
            time_value = time_value.to_pydatetime()
        output = {"time": time_value,
                  "distance": {"data": functions.filter_variable(sp_arr), "unit": "m",
                               "description": "Distance along transect"},
                  "depth": {"data": functions.filter_variable(depth), "unit": "m",
                            "description": "Distance from the surface"},
                  "lat": functions.filter_variable(lat_arr, decimals=5),
                  "lng": functions.filter_variable(lng_arr, decimals=5),
                  "variables": {}
                  }
        if "temperature" in variables:
            t = ds.R1.isel(M=xi, N=yi, LSTSCI=0).transpose('KMAXOUT_RESTR', 'dim_0').values
            t[:, ~vd_arr.astype(bool)] = np.nan
            output["variables"]["temperature"] = {"data": functions.filter_variable(t), "unit": "degC",
                                                  "description": "Water temperature"}
        if "velocity" in variables:
            u, v, = functions.rotate_velocity(
                ds.U1.isel(MC=xi, N=yi).transpose('KMAXOUT_RESTR', 'dim_0').values,
                ds.V1.isel(M=xi, NC=yi).transpose('KMAXOUT_RESTR', 'dim_0').values,
                ds.ALFAS.isel(M=xi, N=yi).values[np.newaxis, :])
            u[:, ~vd_arr.astype(bool)] = np.nan
            v[:, ~vd_arr.astype(bool)] = np.nan
            output["variables"]["u"] = {"data": functions.filter_variable(u, decimals=5), "unit": "m/s",
                                        "description": "Eastward flow velocity"}
            output["variables"]["v"] = {"data": functions.filter_variable(u, decimals=5), "unit": "m/s",
                                        "description": "Northward flow velocity"}
    return output


def get_simulations_transect_mitgcm(filesystem, lake, time, latitude_str, longitude_str, variables, nodata=-999.0):
    model = "mitgcm"
    variables = [v.lower() for v in variables]
    y_list = [float(x) for x in latitude_str.replace(" ", "").split(",")]
    x_list = [float(x) for x in longitude_str.replace(" ", "").split(",")]

    if len(y_list) < 2 or len(y_list) != len(x_list):
        raise HTTPException(status_code=400,
                            detail="At least two valid points should be provided.")

    origin = datetime.strptime(time, "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
    last_sunday = origin + relativedelta(weekday=SU(-1))
    previous_sunday = last_sunday - timedelta(days=7)
    lakes = os.path.join(filesystem, "media/simulations", model, "results")
    if not os.path.isdir(os.path.join(lakes, lake)):
        raise HTTPException(status_code=400,
                            detail="{} simulation results are not available for {} please select from: [{}]"
                            .format(model, lake, ", ".join(os.listdir(lakes))))
    if os.path.isfile(os.path.join(lakes, lake, "{}.nc".format(last_sunday.strftime("%Y%m%d")))):
        file = os.path.join(lakes, lake, "{}.nc".format(last_sunday.strftime("%Y%m%d")))
    elif os.path.isfile(os.path.join(lakes, lake, "{}.nc".format(previous_sunday.strftime("%Y%m%d")))):
        file = os.path.join(lakes, lake, "{}.nc".format(previous_sunday.strftime("%Y%m%d")))
    else:
        raise HTTPException(status_code=400,
                            detail="Apologies data is not available for {} at {}".format(lake, time))

    with xr.open_mfdataset(file) as ds:
        ds['time'] = ds.indexes['time'].tz_localize('UTC')
        time_index = functions.get_closest_index(functions.convert_to_unit(origin, "nano"), ds["time"].values)
        ds = ds.isel(time=time_index)
        x = ds.lng[:].values
        y = ds.lat[:].values
        z = ds.depth[0, :].values if len(ds.depth.shape) == 2 else ds.depth[:].values
        grid_spacing = functions.center_grid_spacing(x, y)
        indexes = np.where((x >= np.min(x_list) - 2 * grid_spacing) &
                           (x <= np.max(x_list) + 2 * grid_spacing) &
                           (y >= np.min(y_list) - 2 * grid_spacing) &
                           (y <= np.max(y_list) + 2 * grid_spacing))

        start = 0
        xi_arr, yi_arr, sp_arr, vd_arr = np.array([]), np.array([]), np.array([]), np.array([])
        for i in range(len(x_list) - 1):
            xi, yi, sp, vd, distance = functions.line_segments(x_list[i], y_list[i], x_list[i + 1], y_list[i + 1], x, y,
                                                               indexes, start, grid_spacing, yx=True)
            start = start + distance
            xi_arr = np.concatenate((xi_arr, xi), axis=0)
            yi_arr = np.concatenate((yi_arr, yi), axis=0)
            sp_arr = np.concatenate((sp_arr, sp), axis=0)
            vd_arr = np.concatenate((vd_arr, vd), axis=0)

        xi_arr = xi_arr.astype(int)
        yi_arr = yi_arr.astype(int)
        lat_arr, lng_arr = y[yi_arr, xi_arr], x[yi_arr, xi_arr]

        xi = xr.DataArray(xi_arr)
        yi = xr.DataArray(yi_arr)

        t = ds.t.isel(X=xi, Y=yi).values
        valid_depths = ~np.all(np.isnan(t), axis=1)
        depth = z[valid_depths]
        ds = ds.sel(depth=valid_depths)
        time_value = ds.time.values
        if isinstance(time_value, np.datetime64):
            time_value = pd.Timestamp(time_value).tz_localize('UTC')
            time_value = time_value.to_pydatetime()
        output = {"time": time_value,
                  "distance": {"data": functions.filter_variable(sp_arr), "unit": "m",
                               "description": "Distance along transect"},
                  "depth": {"data": functions.filter_variable(depth), "unit": "m",
                            "description": "Distance from the surface"},
                  "lat": functions.filter_variable(lat_arr, decimals=5),
                  "lng": functions.filter_variable(lng_arr, decimals=5),
                  "variables": {}
                  }
        if "temperature" in variables:
            t = ds.t.isel(X=xi, Y=yi).transpose('depth', 'dim_0').values
            t[:, ~vd_arr.astype(bool)] = np.nan
            output["variables"]["temperature"] = {"data": functions.filter_variable(t), "unit": "degC",
                                                  "description": "Water temperature"}
        if "velocity" in variables:
            u = ds.u.isel(X=xi, Y=yi).transpose('depth', 'dim_0').values
            v = ds.v.isel(X=xi, Y=yi).transpose('depth', 'dim_0').values
            u[:, ~vd_arr.astype(bool)] = np.nan
            v[:, ~vd_arr.astype(bool)] = np.nan
            output["variables"]["u"] = {"data": functions.filter_variable(u, decimals=5), "unit": "m/s",
                                        "description": "Eastward flow velocity"}
            output["variables"]["v"] = {"data": functions.filter_variable(u, decimals=5), "unit": "m/s",
                                        "description": "Northward flow velocity"}
    return output


def get_simulations_transect_period(filesystem, model, lake, start, end, latitude_list, longitude_list, variables):
    if model == "delft3d-flow":
        return get_simulations_transect_period_delft3dflow(filesystem, lake, start, end, latitude_list, longitude_list, variables)
    if model == "mitgcm":
        return get_simulations_transect_period_mitgcm(filesystem, lake, start, end, latitude_list, longitude_list, variables)
    else:
        raise HTTPException(status_code=400,
                            detail="Apologies profile extraction not available for {}".format(model))


def get_simulations_transect_period_delft3dflow(filesystem, lake, start, end, latitude_str, longitude_str, variables, nodata=-999.0):
    model = "delft3d-flow"
    variables = [v.lower() for v in variables]
    latitude_list = [float(x) for x in latitude_str.replace(" ", "").split(",")]
    longitude_list = [float(x) for x in longitude_str.replace(" ", "").split(",")]

    if len(latitude_list) < 2 or len(latitude_list) != len(longitude_list):
        raise HTTPException(status_code=400,
                            detail="At least two valid points should be provided.")

    lakes = os.path.join(filesystem, "media/simulations", model, "results")
    if not os.path.isdir(os.path.join(lakes, lake)):
        raise HTTPException(status_code=400,
                            detail="{} simulation results are not available for {} please select from: [{}]"
                            .format(model, lake, ", ".join(os.listdir(lakes))))

    weeks = functions.sundays_between_dates(datetime.strptime(start[0:8], "%Y%m%d").replace(tzinfo=timezone.utc),
                                            datetime.strptime(end[0:8], "%Y%m%d").replace(tzinfo=timezone.utc))
    files = [os.path.join(lakes, lake, "{}.nc".format(week.strftime("%Y%m%d"))) for week in weeks]

    files = [file for file in files if os.path.isfile(file)]
    if len(files) == 0:
        raise HTTPException(status_code=400, detail="Apologies data is not available for your requested period")

    start_datetime = datetime.strptime(start[0:10], "%Y%m%d%H").replace(tzinfo=timezone.utc)
    end_datetime = datetime.strptime(end[0:10], "%Y%m%d%H").replace(tzinfo=timezone.utc)

    with xr.open_mfdataset(files, data_vars='minimal', compat='override', coords='minimal') as ds:
        ds['time'] = ds.indexes['time'].tz_localize('UTC')
        ds = ds.sel(time=slice(start_datetime, end_datetime))
        if len(ds['time']) == 0:
            raise HTTPException(status_code=400,
                                detail="No timesteps available between {} and {}".format(start, end))
        x = ds.XZ[:].values
        y = ds.YZ[:].values
        if len(ds.ZK_LYR.shape) == 2:
            z = ds.ZK_LYR[0, :].values * - 1
        else:
            z = ds.ZK_LYR[:].values * - 1
        grid_spacing = functions.center_grid_spacing(x, y)
        projection = functions.identify_projection(np.max(x), np.max(y))
        if projection == "WGS84":
            raise HTTPException(status_code=400, detail="Method not implemented for models with projection WGS84")
        x_list, y_list = functions.latlng_to_projection(latitude_list, longitude_list, projection)
        indexes = np.where((x >= np.min(x_list) - 2 * grid_spacing) &
                           (x <= np.max(x_list) + 2 * grid_spacing) &
                           (y >= np.min(y_list) - 2 * grid_spacing) &
                           (y <= np.max(y_list) + 2 * grid_spacing))
        start = 0
        xi_arr, yi_arr, sp_arr, vd_arr = np.array([]), np.array([]), np.array([]), np.array([])
        for i in range(len(x_list) - 1):
            xi, yi, sp, vd, distance = functions.line_segments(x_list[i], y_list[i], x_list[i + 1], y_list[i + 1], x, y,
                                                               indexes, start, grid_spacing)
            start = start + distance
            xi_arr = np.concatenate((xi_arr, xi), axis=0)
            yi_arr = np.concatenate((yi_arr, yi), axis=0)
            sp_arr = np.concatenate((sp_arr, sp), axis=0)
            vd_arr = np.concatenate((vd_arr, vd), axis=0)

        xi_arr = xi_arr.astype(int)
        yi_arr = yi_arr.astype(int)
        lat_arr, lng_arr = functions.projection_to_latlng(x[xi_arr, yi_arr], y[xi_arr, yi_arr], projection)

        xi = xr.DataArray(xi_arr)
        yi = xr.DataArray(yi_arr)

        t = ds.R1.isel(time=0, M=xi, N=yi, LSTSCI=0).values
        valid_depths = ~np.all(t == nodata, axis=1)
        depth = z[valid_depths]
        ds = ds.sel(KMAXOUT_RESTR=valid_depths)
        output = {"time": functions.alplakes_time(ds.time[:].values, "nano"),
                  "distance": {"data": functions.filter_variable(sp_arr), "unit": "m",
                               "description": "Distance along transect"},
                  "depth": {"data": functions.filter_variable(depth), "unit": "m",
                           "description": "Distance from the surface"},
                  "lat": functions.filter_variable(lat_arr, decimals=5),
                  "lng": functions.filter_variable(lng_arr, decimals=5),
                  "variables": {}
                  }
        if "temperature" in variables:
            t = ds.R1.isel(M=xi, N=yi, LSTSCI=0).transpose('time', 'KMAXOUT_RESTR', 'dim_0').values
            t[:, :, ~vd_arr.astype(bool)] = np.nan
            output["variables"]["temperature"] = {"data": functions.filter_variable(t), "unit": "degC", "description": "Water temperature"}
        if "velocity" in variables:
            if "time" in ds.ALFAS.dims:
                alfas = ds.ALFAS.isel(M=xi, N=yi).transpose('time', 'dim_0').values[:, np.newaxis, :]
            else:
                alfas = ds.ALFAS.isel(M=xi, N=yi).values
            u, v, = functions.rotate_velocity(
                ds.U1.isel(MC=xi, N=yi).transpose('time', 'KMAXOUT_RESTR', 'dim_0').values,
                ds.V1.isel(M=xi, NC=yi).transpose('time', 'KMAXOUT_RESTR', 'dim_0').values,
                alfas)
            u[:, :, ~vd_arr.astype(bool)] = np.nan
            v[:, :, ~vd_arr.astype(bool)] = np.nan
            output["variables"]["u"] = {"data": functions.filter_variable(u, decimals=5), "unit": "m/s", "description": "Eastward flow velocity"}
            output["variables"]["v"] = {"data": functions.filter_variable(u, decimals=5), "unit": "m/s", "description": "Northward flow velocity"}
    return output


def get_simulations_transect_period_mitgcm(filesystem, lake, start, end, latitude_str, longitude_str, variables, nodata=-999.0):
    model = "mitgcm"
    variables = [v.lower() for v in variables]
    y_list = [float(x) for x in latitude_str.replace(" ", "").split(",")]
    x_list = [float(x) for x in longitude_str.replace(" ", "").split(",")]

    if len(y_list) < 2 or len(y_list) != len(x_list):
        raise HTTPException(status_code=400,
                            detail="At least two valid points should be provided.")

    lakes = os.path.join(filesystem, "media/simulations", model, "results")
    if not os.path.isdir(os.path.join(lakes, lake)):
        raise HTTPException(status_code=400,
                            detail="{} simulation results are not available for {} please select from: [{}]"
                            .format(model, lake, ", ".join(os.listdir(lakes))))

    weeks = functions.sundays_between_dates(datetime.strptime(start[0:8], "%Y%m%d").replace(tzinfo=timezone.utc),
                                            datetime.strptime(end[0:8], "%Y%m%d").replace(tzinfo=timezone.utc))
    files = [os.path.join(lakes, lake, "{}.nc".format(week.strftime("%Y%m%d"))) for week in weeks]

    files = [file for file in files if os.path.isfile(file)]
    if len(files) == 0:
        raise HTTPException(status_code=400, detail="Apologies data is not available for your requested period")

    start_datetime = datetime.strptime(start[0:10], "%Y%m%d%H").replace(tzinfo=timezone.utc)
    end_datetime = datetime.strptime(end[0:10], "%Y%m%d%H").replace(tzinfo=timezone.utc)

    with xr.open_mfdataset(files, data_vars='minimal', compat='override', coords='minimal') as ds:
        ds['time'] = ds.indexes['time'].tz_localize('UTC')
        ds = ds.sel(time=slice(start_datetime, end_datetime))
        if len(ds['time']) == 0:
            raise HTTPException(status_code=400,
                                detail="No timesteps available between {} and {}".format(start, end))
        x = ds.lng[0, :].values if len(ds.lng.shape) == 3 else ds.lng[:].values
        y = ds.lat[0, :].values if len(ds.lat.shape) == 3 else ds.lat[:].values
        z = ds.depth[0, :].values if len(ds.depth.shape) == 2 else ds.depth[:].values
        grid_spacing = functions.center_grid_spacing(x, y)
        indexes = np.where((x >= np.min(x_list) - 2 * grid_spacing) &
                           (x <= np.max(x_list) + 2 * grid_spacing) &
                           (y >= np.min(y_list) - 2 * grid_spacing) &
                           (y <= np.max(y_list) + 2 * grid_spacing))
        start = 0
        xi_arr, yi_arr, sp_arr, vd_arr = np.array([]), np.array([]), np.array([]), np.array([])
        for i in range(len(x_list) - 1):
            xi, yi, sp, vd, distance = functions.line_segments(x_list[i], y_list[i], x_list[i + 1], y_list[i + 1], x, y,
                                                               indexes, start, grid_spacing, yx=True)
            start = start + distance
            xi_arr = np.concatenate((xi_arr, xi), axis=0)
            yi_arr = np.concatenate((yi_arr, yi), axis=0)
            sp_arr = np.concatenate((sp_arr, sp), axis=0)
            vd_arr = np.concatenate((vd_arr, vd), axis=0)

        xi_arr = xi_arr.astype(int)
        yi_arr = yi_arr.astype(int)
        lat_arr, lng_arr = y[yi_arr, xi_arr], x[yi_arr, xi_arr]

        xi = xr.DataArray(xi_arr)
        yi = xr.DataArray(yi_arr)

        t = ds.t.isel(time=0, X=xi, Y=yi).values
        valid_depths = ~np.all(np.isnan(t), axis=1)
        depth = z[valid_depths]
        ds = ds.sel(depth=valid_depths)
        output = {"time": functions.alplakes_time(ds.time[:].values, "nano"),
                  "distance": {"data": functions.filter_variable(sp_arr), "unit": "m",
                               "description": "Distance along transect"},
                  "depth": {"data": functions.filter_variable(depth), "unit": "m",
                           "description": "Distance from the surface"},
                  "lat": functions.filter_variable(lat_arr, decimals=5),
                  "lng": functions.filter_variable(lng_arr, decimals=5),
                  "variables": {}
                  }
        if "temperature" in variables:
            t = ds.t.isel(X=xi, Y=yi).transpose('time', 'depth', 'dim_0').values
            t[:, :, ~vd_arr.astype(bool)] = np.nan
            output["variables"]["temperature"] = {"data": functions.filter_variable(t), "unit": "degC", "description": "Water temperature"}
        if "velocity" in variables:
            u = ds.u.isel(X=xi, Y=yi).transpose('time', 'depth', 'dim_0').values
            v = ds.v.isel(X=xi, Y=yi).transpose('time', 'depth', 'dim_0').values
            u[:, :, ~vd_arr.astype(bool)] = np.nan
            v[:, :, ~vd_arr.astype(bool)] = np.nan
            output["variables"]["u"] = {"data": functions.filter_variable(u, decimals=5), "unit": "m/s", "description": "Eastward flow velocity"}
            output["variables"]["v"] = {"data": functions.filter_variable(u, decimals=5), "unit": "m/s", "description": "Northward flow velocity"}
    return output


class OneDimensionalModels(str, Enum):
    simstrat = "simstrat"


class SimstratResampleOptions(str, Enum):
    daily = "daily"
    monthly = "monthly"
    yearly = "yearly"

class MetadataKeyModel1D(BaseModel):
    unit: Union[str, None] = None
    description: Union[str, None] = None
    dimensions: List[str]

class MetadataLake1DDetail(BaseModel):
    name: str
    depth: List[float]
    start_date: date
    end_date: date
    missing_dates: List[date]
    variables: Dict[str, MetadataKeyModel1D]

class Metadata1D(BaseModel):
    model: str
    lakes: List[MetadataLake1DDetail]

class ResponseModel1DPoint(BaseModel):
    time: List[datetime]
    depth: functions.VariableKeyModel1D
    resample: Union[str, None]
    variables: Dict[str, functions.VariableKeyModel1D]
    @field_validator('time')
    @classmethod
    def validate_timezone(cls, value):
        if isinstance(value, list):
            for v in value:
                if v.tzinfo is None:
                    raise ValueError('time must have a timezone')
        elif value.tzinfo is None:
            raise ValueError('time must have a timezone')
        return value

class ResponseModel1DProfile(BaseModel):
    time: datetime
    depth: functions.VariableKeyModel1D
    variables: Dict[str, functions.VariableKeyModel1D]
    @field_validator('time')
    @classmethod
    def validate_timezone(cls, value):
        if isinstance(value, list):
            for v in value:
                if v.tzinfo is None:
                    raise ValueError('time must have a timezone')
        elif value.tzinfo is None:
            raise ValueError('time must have a timezone')
        return value

class ResponseModel1DDepthTime(BaseModel):
    time: List[datetime]
    depth: functions.VariableKeyModel1D
    variables: Dict[str, functions.VariableKeyModel1D]
    @field_validator('time')
    @classmethod
    def validate_timezone(cls, value):
        if isinstance(value, list):
            for v in value:
                if v.tzinfo is None:
                    raise ValueError('time must have a timezone')
        elif value.tzinfo is None:
            raise ValueError('time must have a timezone')
        return value

class ProductDOY(BaseModel):
    depth: str
    variable: str

class MetadataLake1DDOY(BaseModel):
    name: str
    products: List[ProductDOY]

class Metadata1DDOY(BaseModel):
    model: str
    lakes: List[MetadataLake1DDOY]

class ResponseModel1DDOY(BaseModel):
    start_time: datetime
    end_time: datetime
    depth: functions.VariableKeyModel1D
    variables: Dict[str, functions.VariableKeyModel1D]
    @field_validator('start_time', 'end_time')
    @classmethod
    def validate_timezone(cls, value):
        if isinstance(value, list):
            for v in value:
                if v.tzinfo is None:
                    raise ValueError('time must have a timezone')
        elif value.tzinfo is None:
            raise ValueError('time must have a timezone')
        return value


def get_one_dimensional_metadata(filesystem):
    metadata = []
    models = os.listdir(os.path.join(filesystem, "media/1dsimulations"))

    for model in models:
        lakes = os.listdir(os.path.join(filesystem, "media/1dsimulations", model, "results"))
        m = {"model": model, "lakes": []}

        for lake in lakes:
            if model == "simstrat":
                path = os.path.join(os.path.join(filesystem, "media/1dsimulations", model, "results", lake))
                files = os.listdir(path)
                files = [file for file in files if len(file.split(".")[0]) == 6 and file.split(".")[1] == "nc"]
                files.sort()
                combined = '_'.join(files)
                missing_dates = []

                with netCDF4.Dataset(os.path.join(path, files[0])) as nc:
                    start_date = functions.convert_from_unit(nc.variables["time"][0], nc.variables["time"].units)

                with netCDF4.Dataset(os.path.join(path, files[-1])) as nc:
                    end_date = functions.convert_from_unit(nc.variables["time"][-1], nc.variables["time"].units)
                    depths = np.array(nc.variables["depth"][:]) * -1
                    depths = depths[depths > 0]
                    depths.sort()
                    depths = [float("%.2f" % d) for d in depths]

                    variables = {}
                    for var in nc.variables:
                        if var not in ["time", "depth"]:
                            long_name = nc.variables[var].long_name if 'long_name' in nc.variables[
                                var].ncattrs() else var
                            variables[var] = {
                                "description": long_name,
                                "unit": nc.variables[var].units,
                                "dimensions": nc.variables[var].dimensions
                            }

                for d in functions.monthrange(start_date, end_date, months=1):
                    if d.strftime('%Y%m') not in combined:
                        missing_dates.append(d.strftime("%Y-%m"))

                m["lakes"].append({"name": lake,
                                   "depth": depths,
                                   "start_date": start_date.strftime("%Y-%m-%d"),
                                   "end_date": end_date.strftime("%Y-%m-%d"),
                                   "missing_dates": missing_dates,
                                   "variables": variables})
            else:
                print("Model not recognised.")
        metadata.append(m)
    return metadata


def get_one_dimensional_metadata_lake(filesystem, model, lake):
    path = os.path.join(os.path.join(filesystem, "media/1dsimulations", model, "results", lake))
    files = os.listdir(path)
    files = [file for file in files if len(file.split(".")[0]) == 6 and file.split(".")[1] == "nc"]
    files.sort()
    combined = '_'.join(files)
    missing_dates = []

    out = {"name": lake}

    if model == "simstrat":
        with netCDF4.Dataset(os.path.join(path, files[0])) as nc:
            start_date = functions.convert_from_unit(nc.variables["time"][0], nc.variables["time"].units)
            out["start_date"] = start_date.strftime("%Y-%m-%d")

        with netCDF4.Dataset(os.path.join(path, files[-1])) as nc:
            end_date = functions.convert_from_unit(nc.variables["time"][-1], nc.variables["time"].units)
            out["end_date"] = end_date.strftime("%Y-%m-%d")
            depths = np.array(nc.variables["depth"][:]) * -1
            depths = depths[depths > 0]
            depths.sort()
            depths = [float("%.2f" % d) for d in depths]
            out["depth"] = depths

            variables = {}
            for var in nc.variables:
                if var not in ["time", "depth"]:
                    long_name = nc.variables[var].long_name if 'long_name' in nc.variables[var].ncattrs() else var
                    variables[var] = {
                        "description": long_name,
                        "unit": nc.variables[var].units,
                        "dimensions": nc.variables[var].dimensions
                    }
            out["variables"] = variables

        for d in functions.monthrange(start_date, end_date, months=1):
            if d.strftime('%Y%m') not in combined:
                missing_dates.append([d.strftime("%Y-%m"), (d + timedelta(days=7)).strftime("%Y-%m")])
        out["missing_dates"] = missing_dates

        return out
    else:
        raise ValueError("Model not recognised.")


def get_one_dimensional_file(filesystem, model, lake, month):
    path = os.path.join(filesystem, "media/1dsimulations", model, "results", lake, "{}.nc".format(month))
    if os.path.isfile(path):
        return FileResponse(path, media_type="application/nc", filename="{}_{}_{}.nc".format(model, lake, month))
    else:
        raise HTTPException(status_code=400,
                            detail="Apologies data is not available for {} on the month beginning {}".format(model,
                                                                                                             month))


def get_one_dimensional_point(filesystem, model, lake, start_time, end_time, depth, variables, resample=None):
    if model == "simstrat":
        return get_one_dimensional_point_simstrat(filesystem, lake, start_time, end_time, depth, variables, resample)
    else:
        raise HTTPException(status_code=400,
                            detail="Apologies not available for {}".format(model))


def get_one_dimensional_point_simstrat(filesystem, lake, start, end, depth, variables, resample):
    model = "simstrat"
    out = {"variables": {}}
    lakes = os.path.join(filesystem, "media/1dsimulations", model, "results")
    if not os.path.isdir(os.path.join(lakes, lake)):
        raise HTTPException(status_code=400,
                            detail="{} simulation results are not available for {} please select from: [{}]"
                            .format(model, lake, ", ".join(os.listdir(lakes))))
    months = functions.months_between_dates(datetime.strptime(start[0:8], "%Y%m%d").replace(tzinfo=timezone.utc),
                                            datetime.strptime(end[0:8], "%Y%m%d").replace(tzinfo=timezone.utc))
    files = []
    for month in months:
        file = os.path.join(lakes, lake, "{}.nc".format(month.strftime("%Y%m")))
        if os.path.isfile(file):
            files.append(file)

    if len(files) == 0:
        raise HTTPException(status_code=400,
                            detail="Apologies data is not available for requested period")

    start_datetime = datetime.strptime(start, "%Y%m%d%H%M").replace(tzinfo=timezone.utc).astimezone().replace(tzinfo=None)
    end_datetime = datetime.strptime(end, "%Y%m%d%H%M").replace(tzinfo=timezone.utc).astimezone().replace(tzinfo=None)

    with xr.open_mfdataset(files, data_vars='minimal', compat='override', coords='minimal') as ds:
        dims = None
        for v in variables:
            if v not in ds.variables:
                raise HTTPException(status_code=400, detail="Variable {} is not available".format(v))
            if dims is not None and len(ds[v].shape) != dims:
                raise HTTPException(status_code=400, detail="Variables do not have consistent dimensions".format(v))
            dims = len(ds[v].shape)
        ds = ds.sel(time=slice(start_datetime, end_datetime))
        if len(ds['time']) == 0:
            raise HTTPException(status_code=400,
                                detail="No timesteps available between {} and {}".format(start, end))
        out["depth"] = {"data": None, "unit": "m", "description": "Distance from the surface to the closest grid point to requested depth"}
        if len(ds[variables[0]].shape) == 2:
            depths = ds.depth[:].values * - 1
            index = functions.get_closest_index(depth, depths)
            out["depth"]["data"] = depths[index]
            ds = ds.sel(depth=depths[index] * -1)

        df_dict = {'time': pd.to_datetime(ds['time'].values).tz_localize('UTC')}
        for v in variables:
            df_dict[v] = ds[v][:].values
        df = pd.DataFrame(df_dict)
        resample_options = {"hourly": "H", "daily": "D", "monthly": "M", "yearly": "Y"}
        if resample is not None:
            df.set_index('time', inplace=True)
            df = df.resample(resample_options[resample], label='left').mean(numeric_only=True)
            df = df.reset_index()
            out["resample"] = resample + " mean"
        else:
            out["resample"] = None
        out["time"] = [x.replace(tzinfo=timezone.utc) for x in df["time"].tolist()]
        for v in variables:
            out["variables"][v] = {"data": functions.filter_variable(df[v]), "unit": ds[v].units, "description": ds[v].long_name}
        return out


def get_one_dimensional_profile(filesystem, model, lake, time, variables):
    if model == "simstrat":
        return get_one_dimensional_profile_simstrat(filesystem, lake, time, variables)
    else:
        raise HTTPException(status_code=400, detail="Apologies not available for {}".format(model))


def get_one_dimensional_profile_simstrat(filesystem, lake, time, variables):
    model = "simstrat"
    out = {"variables":{}}
    lakes = os.path.join(filesystem, "media/1dsimulations", model, "results")
    origin = datetime.strptime(time, "%Y%m%d%H%M").replace(tzinfo=timezone.utc)

    if not os.path.isdir(os.path.join(lakes, lake)):
        raise HTTPException(status_code=400,
                            detail="{} simulation results are not available for {} please select from: [{}]"
                            .format(model, lake, ", ".join(os.listdir(lakes))))
    file = os.path.join(lakes, lake, "{}.nc".format(origin.strftime("%Y%m")))
    if not os.path.isfile(file):
            raise HTTPException(status_code=400,
                                detail="Apologies data is not available for {} month {}".format(lake,
                                                                                                origin.strftime("%Y%m")))
    with xr.open_mfdataset(file) as ds:
        for v in variables:
            if v not in ds.variables:
                raise HTTPException(status_code=400, detail="Variable {} is not available".format(v))
            if len(ds[v].shape) == 1:
                raise HTTPException(status_code=400, detail="Variable {} exists but is not 2D".format(v))
        ds['time'] = ds.indexes['time'].tz_localize('UTC')
        time_index = functions.get_closest_index(functions.convert_to_unit(origin, "nano"), ds["time"].values)
        ds = ds.isel(time=time_index)
        depths = ds.depth[:].values * - 1
        time_value = ds.time.values
        if isinstance(time_value, np.datetime64):
            time_value = pd.Timestamp(time_value).tz_localize('UTC')
            time_value = time_value.to_pydatetime()
        out["time"] = time_value
        out["depth"] = {"data": functions.filter_variable(depths), "unit": "m", "description": "Distance from the surface to the closest grid point to requested depth"}
        for v in variables:
            out["variables"][v] = {"data": functions.filter_variable(ds[v].values), "unit": ds[v].units, "description": ds[v].long_name}
        return out


def get_one_dimensional_depth_time(filesystem, model, lake, start_time, end_time, variables):
    if model == "simstrat":
        return get_one_dimensional_depth_time_simstrat(filesystem, lake, start_time, end_time, variables)
    else:
        raise HTTPException(status_code=400, detail="Apologies not available for {}".format(model))


def get_one_dimensional_depth_time_simstrat(filesystem, lake, start, end, variables):
    model = "simstrat"
    out = {"variables":{}}
    lakes = os.path.join(filesystem, "media/1dsimulations", model, "results")
    if not os.path.isdir(os.path.join(lakes, lake)):
        raise HTTPException(status_code=400,
                            detail="{} simulation results are not available for {} please select from: [{}]"
                            .format(model, lake, ", ".join(os.listdir(lakes))))
    months = functions.months_between_dates(datetime.strptime(start[0:8], "%Y%m%d").replace(tzinfo=timezone.utc),
                                            datetime.strptime(end[0:8], "%Y%m%d").replace(tzinfo=timezone.utc))

    for month in months:
        if not os.path.isfile(os.path.join(lakes, lake, "{}.nc".format(month.strftime("%Y%m")))):
            raise HTTPException(status_code=400,
                                detail="Apologies data is not available for {} month {}".format(lake,
                                                                                                month.strftime("%Y%m")))

    start_datetime = datetime.strptime(start, "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
    end_datetime = datetime.strptime(end, "%Y%m%d%H%M").replace(tzinfo=timezone.utc)

    files = [os.path.join(lakes, lake, "{}.nc".format(month.strftime("%Y%m"))) for month in months]
    with xr.open_mfdataset(files, data_vars='minimal', compat='override', coords='minimal') as ds:
        for v in variables:
            if v not in ds.variables:
                raise HTTPException(status_code=400, detail="Variable {} is not available".format(v))
            if len(ds[v].shape) == 1:
                raise HTTPException(status_code=400, detail="Variable {} exists but is not 2D".format(v))
        ds['time'] = ds.indexes['time'].tz_localize('UTC')
        ds = ds.sel(time=slice(start_datetime, end_datetime))
        if len(ds['time']) == 0:
            raise HTTPException(status_code=400,
                                detail="No timesteps available between {} and {}".format(start, end))
        depths = ds.depth[:].values * - 1
        out["time"] = functions.alplakes_time(ds.time[:].values, "nano")
        out["depth"] = {"data": functions.filter_variable(depths), "unit": "m", "description": "Distance from the surface to the closest grid point to requested depth"}
        for v in variables:
            out["variables"][v] = {"data": functions.filter_variable(ds[v][:].values), "unit": ds[v].units, "description": ds[v].long_name}
        return out


def get_one_dimensional_day_of_year_metadata(filesystem):
    base_path = os.path.join(filesystem, "media/1dsimulations")
    output = []
    for model in os.listdir(base_path):
        if os.path.exists(os.path.join(base_path, model, "doy")):
            products = os.listdir(os.path.join(base_path, model, "doy"))
            if len(products) > 0:
                lakes = set([p.replace(".json", "").split("_")[0] for p in products])
                lakes_dict = {l: [] for l in lakes}
                for product in products:
                    product_split = product.replace(".json", "").split("_")
                    lake = product_split[0]
                    variable = product_split[1]
                    depth = product_split[2]
                    lakes_dict[lake].append({"depth": depth, "variable": variable})
                output.append({
                    "model": model,
                    "lakes": [{"name": l, "products": lakes_dict[l]} for l in lakes]
                })
    return output


def get_one_dimensional_day_of_year(filesystem, model, lake, variable, depth):
    if model == "simstrat":
        return get_one_dimensional_day_of_year_simstrat(filesystem, lake, variable, depth)
    else:
        raise HTTPException(status_code=400, detail="Apologies not available for {}".format(model))


def get_one_dimensional_day_of_year_simstrat(filesystem, lake, variable, depth):
    model = "simstrat"
    lakes = os.path.join(filesystem, "media/1dsimulations", model, "results")
    if not os.path.isdir(os.path.join(lakes, lake)):
        raise HTTPException(status_code=400,
                            detail="{} simulation results are not available for {} please select from: [{}]"
                            .format(model, lake, ", ".join(os.listdir(lakes))))

    doy_files = os.listdir(os.path.join(filesystem, "media/1dsimulations", model, "doy"))
    filtered_doy_files = [f for f in doy_files
                          if "{}_{}_".format(lake, variable) in f
                          and abs(float(f.replace(".json", "").split("_")[-1]) - float(depth)) < 0.1]

    if len(filtered_doy_files) > 0:
        with open(os.path.join(filesystem, "media/1dsimulations", model, "doy", filtered_doy_files[0]), "r") as f:
            out = json.load(f)
        return out
    else:
        raise HTTPException(status_code=400, detail="Apologies DOY has not been computed for your request.")


def write_one_dimensional_day_of_year(filesystem, model, lake, variable, depth):
    if model == "simstrat":
        return write_one_dimensional_day_of_year_simstrat(filesystem, lake, variable, depth)
    else:
        raise HTTPException(status_code=400, detail="Apologies not available for {}".format(model))


def write_one_dimensional_day_of_year_simstrat(filesystem, lake, variable, depth):
    model = "simstrat"
    lakes = os.path.join(filesystem, "media/1dsimulations", model, "results")
    if not os.path.isdir(os.path.join(lakes, lake)):
        raise HTTPException(status_code=400,
                            detail="{} simulation results are not available for {} please select from: [{}]"
                            .format(model, lake, ", ".join(os.listdir(lakes))))
    files = [os.path.join(lakes, lake, file) for file in os.listdir(os.path.join(lakes, lake)) if file.endswith(".nc")]
    files.sort()
    if len(files) > 48:
        files = files[24:]  # Remove first two years as a warmup
    try:
        with xr.open_mfdataset(files, data_vars='minimal', compat='override', coords='minimal') as ds:
            if variable not in ds.variables:
                raise HTTPException(status_code=400, detail="Variable {} is not available".format(variable))
            ds['time'] = ds.indexes['time'].tz_localize('UTC')
            depths = ds.depth.values * -1
            depth_index = functions.get_closest_index(depth, depths)
            ds = ds.isel(depth=depth_index)
            df = pd.DataFrame({'time': ds.time.values, 'value': ds[variable].values})
        last_year = pd.Timestamp.now().year - 1
        df["time"] = pd.to_datetime(df['time'], unit='ns', utc=True)
        df.set_index('time', inplace=True)
        df = df[:f'{last_year}-12-31']
        grouped = df.groupby(df.index.dayofyear)['value']
        max_values_doy = grouped.max()
        mean_values_doy = grouped.mean()
        min_values_doy = grouped.min()
        std_values_doy = grouped.std()
        percentile_5_doy = grouped.quantile(0.05)
        percentile_25_doy = grouped.quantile(0.25)
        percentile_75_doy = grouped.quantile(0.75)
        percentile_95_doy = grouped.quantile(0.95)
        df_previous_year = df[f'{last_year}-01-01':f'{last_year}-12-31']
        daily_average_last_year = df_previous_year.groupby(df_previous_year.index.dayofyear)['value'].mean()
        output = {
            "start_time": df.index.min().isoformat(),
            "end_time": df.index.max().isoformat(),
            "depth": {"data": depths[depth_index], "unit": "m", "description": "Distance from the surface"},
            "variables": {
                "doy": {"data": list(range(1, 367)), "unit": ds[variable].units,
                        "description": "Numeric day of year (leap year) where 1 = January 1st"},
                "mean": {"data": functions.filter_variable(mean_values_doy.values), "unit": ds[variable].units,
                         "description": "Mean daily value across reference period"},
                "max": {"data": functions.filter_variable(max_values_doy.values), "unit": ds[variable].units,
                        "description": "Max daily value across reference period"},
                "min": {"data": functions.filter_variable(min_values_doy.values), "unit": ds[variable].units,
                        "description": "Min daily value across reference period"},
                "std": {"data": functions.filter_variable(std_values_doy.values), "unit": ds[variable].units,
                        "description": "Standard deviation of daily value across reference period"},
                "perc5": {"data": functions.filter_variable(percentile_5_doy.values), "unit": ds[variable].units,
                          "description": "5 percent quantile of daily value across reference period"},
                "perc25": {"data": functions.filter_variable(percentile_25_doy.values), "unit": ds[variable].units,
                           "description": "25 percent quantile of daily value across reference period"},
                "perc75": {"data": functions.filter_variable(percentile_75_doy.values), "unit": ds[variable].units,
                           "description": "75 percent quantile of daily value across reference period"},
                "perc95": {"data": functions.filter_variable(percentile_95_doy.values), "unit": ds[variable].units,
                           "description": "95 percent quantile of daily value across reference period"},
                "lastyear": {"data": functions.filter_variable(daily_average_last_year.values),
                             "unit": ds[variable].units, "description": "Mean daily value for the last year only"}
            }
        }
        doy_file = os.path.join(filesystem, "media/1dsimulations", model, "doy",
                                "{}_{}_{}.json".format(lake, variable, float(depths[depth_index])))
        os.makedirs(os.path.dirname(doy_file), exist_ok=True)
        with open(doy_file, "w") as f:
            json.dump(output, f)
        print("Succeeded in producing DOY for {}".format(lake))
    except Exception as e:
        print(e)
        print("Failed to produce DOY for {}".format(lake))
        raise HTTPException(status_code=400, detail="Failed to produce DOY for {}_{}_{}".format(lake, variable, float(depths[depth_index])))
