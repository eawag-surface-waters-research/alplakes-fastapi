import os
import json
import netCDF4
import numpy as np
import xarray as xr
import pandas as pd
from enum import Enum
from typing import Dict, List, Union, Any
from pydantic import BaseModel, validator
from fastapi import HTTPException
from fastapi.responses import FileResponse
from datetime import datetime, timedelta, timezone, date
from dateutil.relativedelta import relativedelta, SU

import matplotlib.pyplot as plt

from app import functions

class MetadataLake(BaseModel):
    name: str
    depths: List[float]
    start_date: date
    end_date: date
    missing_dates: List[date]
    height: int
    width: int

class Metadata(BaseModel):
    model: str
    lakes: List[MetadataLake]

class ResponseModel1D(BaseModel):
    time: List[datetime]
    lat: float
    lng: float
    depth: functions.VariableKeyModel1D
    distance: functions.VariableKeyModel1D
    variables: Dict[str, functions.VariableKeyModel1D]
    @validator('time', each_item=True)
    def validate_timezone(cls, value):
        if value.tzinfo is None:
            raise ValueError('time must have a timezone')
        return value


class ResponseModelLayer(BaseModel):
    time: datetime
    lat: List[List[Any]]
    lng: List[List[Any]]
    depth: functions.VariableKeyModel1D
    variables: Dict[str, functions.VariableKeyModel2D]
    @validator('time', each_item=True)
    def validate_timezone(cls, value):
        if value.tzinfo is None:
            raise ValueError('time must have a timezone')
        return value


class ResponseModelAverageLayer(BaseModel):
    time: List[datetime]
    depth: functions.VariableKeyModel1D
    variables: Dict[str, functions.VariableKeyModel1D]
    @validator('time', each_item=True)
    def validate_timezone(cls, value):
        if value.tzinfo is None:
            raise ValueError('time must have a timezone')
        return value


class ResponseModelProfile(BaseModel):
    time: datetime
    lat: float
    lng: float
    depth: functions.VariableKeyModel1D
    variables: Dict[str, functions.VariableKeyModel1D]
    @validator('time', each_item=True)
    def validate_timezone(cls, value):
        if value.tzinfo is None:
            raise ValueError('time must have a timezone')
        return value

class ResponseModelDepthTime(BaseModel):
    time: List[datetime]
    lat: float
    lng: float
    depth: functions.VariableKeyModel1D
    distance: functions.VariableKeyModel1D
    variables: Dict[str, functions.VariableKeyModel1D]
    @validator('time', each_item=True)
    def validate_timezone(cls, value):
        if value.tzinfo is None:
            raise ValueError('time must have a timezone')
        return value

class ResponseModelTransect(BaseModel):
    time: datetime
    lat: List[float]
    lng: List[float]
    depth: functions.VariableKeyModel1D
    distance: functions.VariableKeyModel1D
    variables: Dict[str, functions.VariableKeyModel1D]
    @validator('time', each_item=True)
    def validate_timezone(cls, value):
        if value.tzinfo is None:
            raise ValueError('time must have a timezone')
        return value

class ResponseModelTransectPeriod(BaseModel):
    time: List[datetime]
    lat: List[float]
    lng: List[float]
    depth: functions.VariableKeyModel1D
    distance: functions.VariableKeyModel1D
    variables: Dict[str, functions.VariableKeyModel1D]
    @validator('time', each_item=True)
    def validate_timezone(cls, value):
        if value.tzinfo is None:
            raise ValueError('time must have a timezone')
        return value


def get_metadata(filesystem):
    metadata = []
    models = os.listdir(os.path.join(filesystem, "media/simulations"))

    for model in models:
        lakes = os.listdir(os.path.join(filesystem, "media/simulations", model, "results"))
        m = {"model": model, "lakes": []}

        for lake in lakes:
            try:
                if model == "delft3d-flow":
                    path = os.path.join(os.path.join(filesystem, "media/simulations", model, "results", lake))
                    files = os.listdir(path)
                    files = [file for file in files if len(file.split(".")[0]) == 8 and file.split(".")[1] == "nc"]
                    files.sort()
                    combined = '_'.join(files)
                    missing_dates = []

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

                    for d in functions.daterange(start_date, end_date, days=7):
                        if d.strftime('%Y%m%d') not in combined:
                            missing_dates.append(d.strftime("%Y-%m-%d"))

                    m["lakes"].append({"name": lake,
                                       "depths": depths,
                                       "start_date": start_date.strftime("%Y-%m-%d"),
                                       "end_date": end_date.strftime("%Y-%m-%d"),
                                       "missing_dates": missing_dates,
                                       "height": height,
                                       "width": width})
                else:
                    raise ValueError("Model not recognised.")
            except:
                print("Failed for {}".format(lake))
        metadata.append(m)
    return metadata


def get_metadata_lake(filesystem, model, lake):
    path = os.path.join(os.path.join(filesystem, "media/simulations", model, "results", lake))
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

        for d in functions.daterange(start_date, end_date, days=7):
            if d.strftime('%Y%m%d') not in combined:
                missing_dates.append([d.strftime("%Y-%m-%d"), (d + timedelta(days=7)).strftime("%Y-%m-%d")])

        return {"name": lake,
                "depths": depths,
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "missing_dates": missing_dates,
                "height": height,
                "width": width}
    else:
        raise ValueError("Model not recognised.")


class Models(str, Enum):
    delft3dflow = "delft3d-flow"


class Lakes(str, Enum):
    geneva = "geneva"
    greifensee = "greifensee"
    zurich = "zurich"
    biel = "biel"
    joux = "joux"
    garda = "garda"
    lugano = "lugano"
    murten = "murten"
    hallwil = "hallwil"
    caldonazzo = "caldonazzo"
    ageri = "ageri"
    stmoritz = "stmoritz"


class Parameters(str, Enum):
    temperature = "temperature"
    velocity = "velocity"
    geometry = "geometry"
    thermocline = "thermocline"


def get_simulations_point(filesystem, model, lake, start, end, depth, latitude, longitude, variables):
    if model == "delft3d-flow":
        return get_simulations_point_delft3dflow(filesystem, lake, start, end, depth, latitude, longitude, variables)
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

    for i, file in enumerate(files):
        if not os.path.isfile(file):
            raise HTTPException(status_code=400,
                                detail="Apologies data is not available for {} week starting {}".format(lake, weeks[i]))

    start_datetime = datetime.strptime(start, "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
    end_datetime = datetime.strptime(end, "%Y%m%d%H%M").replace(tzinfo=timezone.utc)

    with xr.open_mfdataset(files) as ds:
        ds['time'] = ds.indexes['time'].tz_localize('UTC')
        ds = ds.sel(time=slice(start_datetime, end_datetime))
        z = ds.ZK_LYR[0, :].values * -1 if len(ds.ZK_LYR.shape) == 2 else ds.ZK_LYR[:].values * -1
        depth_index = functions.get_closest_index(depth, z)
        depth = float(z[depth_index])
        lat_grid, lng_grid = functions.coordinates_to_latlng(ds.XZ[:].values, ds.YZ[:].values)
        x_index, y_index, distance = functions.get_closest_location(latitude, longitude, lat_grid, lng_grid)
        time = functions.alplakes_time(ds.time.values, "nano").tolist()
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
            output["variables"]["temperature"] = {"data": functions.filter_parameter(t), "unit": "degC", "description": "Water temperature"}
        if "velocity" in variables:
            u, v, = functions.rotate_velocity(
                ds.U1.isel(MC=x_index, N=y_index, KMAXOUT_RESTR=depth_index).values,
                ds.V1.isel(M=x_index, NC=y_index, KMAXOUT_RESTR=depth_index).values,
                ds.ALFAS.isel(M=x_index, N=y_index).values)
            output["variables"]["u"] = {"data": functions.filter_parameter(u, decimals=5), "unit": "m/s", "description": "Eastward flow velocity"}
            output["variables"]["v"] = {"data": functions.filter_parameter(v, decimals=5), "unit": "m/s", "description": "Northward flow velocity"}
    return output


def get_simulations_layer(filesystem, model, lake, time, depth, variables):
    if model == "delft3d-flow":
        return get_simulations_layer_delft3dflow(filesystem, lake, time, depth, variables)
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
               "lat": functions.filter_parameter(lat_grid, decimals=5, nodata=np.nan),
               "lng": functions.filter_parameter(lng_grid, decimals=5, nodata=np.nan),
               "variables": {}
               }
        if "temperature" in variables:
            t = functions.filter_parameter(nc.variables["R1"][time_index, 0, depth_index, :])
            output["variables"]["temperature"] = {"data": functions.filter_parameter(t), "unit": "degC",
                                                  "description": "Water temperature"}
        if "velocity" in variables:
            u, v, = functions.rotate_velocity(nc.variables["U1"][time_index, depth_index, :],
                                              nc.variables["V1"][time_index, depth_index, :],
                                              nc.variables["ALFAS"][:])
            output["variables"]["u"] = {"data": functions.filter_parameter(u, decimals=5), "unit": "m/s",
                                        "description": "Eastward flow velocity"}
            output["variables"]["v"] = {"data": functions.filter_parameter(v, decimals=5), "unit": "m/s",
                                        "description": "Northward flow velocity"}
    return output


def get_simulations_layer_alplakes(filesystem, model, lake, parameter, start, end, depth):
    if model == "delft3d-flow":
        return get_simulations_layer_alplakes_delft3dflow(filesystem, lake, parameter, start, end, depth)
    else:
        raise HTTPException(status_code=400,
                            detail="Apologies data is not available for {}".format(model))


def get_simulations_layer_alplakes_delft3dflow(filesystem, lake, parameter, start, end, depth):
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
            if parameter == "geometry":
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

            if parameter == "temperature":
                f = '%0.2f'
                p = functions.alplakes_parameter(
                    nc.variables["R1"][time_index_start:time_index_end, 0, depth_index, :])
            elif parameter == "velocity":
                f = '%0.5f'
                p = functions.alplakes_velocity(
                    nc.variables["U1"][time_index_start:time_index_end, depth_index, :],
                    nc.variables["V1"][time_index_start:time_index_end, depth_index, :],
                    nc.variables["ALFAS"][:])
            elif parameter == "thermocline":
                if "THERMOCLINE" in nc.variables.keys():
                    f = '%0.2f'
                    p = functions.alplakes_parameter(
                        nc.variables["THERMOCLINE"][time_index_start:time_index_end, :])
                else:
                    raise HTTPException(status_code=400,
                                        detail="Thermocline not available for this dataset. Please try another parameter.")
            else:
                raise HTTPException(status_code=400,
                                    detail="Parameter {} not recognised, please select from: [geometry, temperature, "
                                           "velocity, thermocline]".format(parameter))
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

    for i, file in enumerate(files):
        if not os.path.isfile(file):
            raise HTTPException(status_code=400,
                                detail="Apologies data is not available for {} week starting {}".format(lake, weeks[i]))

    start_datetime = datetime.strptime(start[0:10], "%Y%m%d%H").replace(tzinfo=timezone.utc)
    end_datetime = datetime.strptime(end[0:10], "%Y%m%d%H").replace(tzinfo=timezone.utc)

    with xr.open_mfdataset(files) as ds:
        ds['time'] = ds.indexes['time'].tz_localize('UTC')
        ds = ds.sel(time=slice(start_datetime, end_datetime))
        z = ds.ZK_LYR[0, :].values * -1 if len(ds.ZK_LYR.shape) == 2 else ds.ZK_LYR[:].values * -1
        depth_index = functions.get_closest_index(depth, z)
        depth = float(z[depth_index])
        time = functions.alplakes_time(ds.time.values, "nano").tolist()
        t_arr = ds.R1.isel(KMAXOUT_RESTR=depth_index, LSTSCI=0)
        t_arr = t_arr.where(t_arr != nodata, np.nan)
        t = t_arr.mean(dim=['M', 'N'], skipna=True).values
        output = {"time": time,
                  "depth": {"data": depth, "unit": "m",
                            "description": "Distance from the surface to the closest grid point to requested depth"},
                  "variables": {
                      "temperature": {"data": functions.filter_parameter(t), "unit": "degC", "description": "Water temperature"}
                  }
                  }
    return output


def get_simulations_profile(filesystem, model, lake, dt, latitude, longitude, variables):
    if model == "delft3d-flow":
        return get_simulations_profile_delft3dflow(filesystem, lake, dt, latitude, longitude, variables)
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

        t = functions.filter_parameter(nc.variables["R1"][time_index, 0, :, x_index, y_index])
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
                  "depth": {"data": functions.filter_parameter(depth), "unit": "m", "description": "Distance from the surface to the closest grid point to requested depth"},
                  "variables": {}}
        if "temperature" in variables:
            output["variables"]["temperature"] = {"data": t, "unit": "degC", "description": "Water temperature"}
        if "velocity" in variables:
            output["variables"]["u"] = {"data": functions.filter_parameter(u, decimals=5), "unit": nc.variables["U1"].units,
                  "description": "Eastward flow velocity"}
            output["variables"]["v"] = {"data": functions.filter_parameter(v, decimals=5), "unit": nc.variables["V1"].units,
                  "description": "Northward flow velocity"}
    return output


def get_simulations_depthtime(filesystem, model, lake, start, end, latitude, longitude, variables):
    if model == "delft3d-flow":
        return get_simulations_depthtime_delft3dflow(filesystem, lake, start, end, latitude, longitude, variables)
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

    for i, file in enumerate(files):
        if not os.path.isfile(file):
            raise HTTPException(status_code=400,
                                detail="Apologies data is not available for {} week starting {}".format(lake, weeks[i]))

    start_datetime = datetime.strptime(start, "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
    end_datetime = datetime.strptime(end, "%Y%m%d%H%M").replace(tzinfo=timezone.utc)

    with xr.open_mfdataset(files) as ds:
        ds['time'] = ds.indexes['time'].tz_localize('UTC')
        ds = ds.sel(time=slice(start_datetime, end_datetime))
        lat_grid, lng_grid = functions.coordinates_to_latlng(ds.XZ[:].values, ds.YZ[:].values)
        x_index, y_index, distance = functions.get_closest_location(latitude, longitude, lat_grid, lng_grid)
        t = ds.R1.isel(time=0, M=x_index, N=y_index, LSTSCI=0).values
        depth = ds.ZK_LYR[0, :].values * -1 if len(ds.ZK_LYR.shape) == 2 else ds.ZK_LYR[:].values * -1
        valid_depths = t != nodata
        depth = depth[valid_depths]
        ds = ds.sel(KMAXOUT_RESTR=valid_depths)
        time = functions.alplakes_time(ds.time.values, "nano").tolist()
        output = {"time": time,
                  "lat": lat_grid[x_index, y_index],
                  "lng": lng_grid[x_index, y_index],
                  "depth": {"data": functions.filter_parameter(depth), "unit": "m",
                            "description": "Distance from the surface"},
                  "distance": {"data": distance, "unit": "m",
                               "description": "Distance from requested location to center of closest grid point"},
                  "variables": {}
                  }
        if "temperature" in variables:
            t = ds.R1.isel(M=x_index, N=y_index, LSTSCI=0).transpose('KMAXOUT_RESTR', 'time').values
            output["variables"]["temperature"] = {"data": functions.filter_parameter(t), "unit": "degC",
                                                  "description": "Water temperature"}
        if "velocity" in variables:
            u, v, = functions.rotate_velocity(
                ds.U1.isel(MC=x_index, N=y_index).transpose('KMAXOUT_RESTR', 'time').values,
                ds.V1.isel(M=x_index, NC=y_index).transpose('KMAXOUT_RESTR', 'time').values,
                ds.ALFAS.isel(M=x_index, N=y_index).values[np.newaxis, :])
            output["variables"]["u"] = {"data": functions.filter_parameter(u, decimals=5), "unit": "m/s",
                                        "description": "Eastward flow velocity"}
            output["variables"]["v"] = {"data": functions.filter_parameter(v, decimals=5), "unit": "m/s",
                                        "description": "Northward flow velocity"}
    return output


def get_simulations_transect(filesystem, model, lake, dt, latitude_list, longitude_list, variables):
    if model == "delft3d-flow":
        return get_simulations_transect_delft3dflow(filesystem, lake, dt, latitude_list, longitude_list, variables)
    else:
        raise HTTPException(status_code=400,
                            detail="Apologies profile extraction not available for {}".format(model))


def get_simulations_transect_delft3dflow(filesystem, lake, time, latitude_str, longitude_str, variables, nodata=-999.0):
    model = "delft3d-flow"
    variables = [v.lower() for v in variables]
    latitude_list = [float(x) for x in latitude_str.replace(" ", "").split(",")]
    longitude_list = [float(x) for x in longitude_str.replace(" ", "").split(",")]

    if len(latitude_list) != len(longitude_list):
        raise HTTPException(status_code=400,
                            detail="Latitude list and longitude list are not the same length.")

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
        output = {"time": ds.time.values,
                  "distance": {"data": functions.filter_parameter(sp_arr), "unit": "m",
                               "description": "Distance along transect"},
                  "depth": {"data": functions.filter_parameter(depth), "unit": "m",
                            "description": "Distance from the surface"},
                  "lat": functions.filter_parameter(lat_arr, decimals=5),
                  "lng": functions.filter_parameter(lng_arr, decimals=5),
                  "variables": {}
                  }
        if "temperature" in variables:
            t = ds.R1.isel(M=xi, N=yi, LSTSCI=0).transpose('KMAXOUT_RESTR', 'dim_0').values
            output["variables"]["temperature"] = {"data": functions.filter_parameter(t), "unit": "degC",
                                                  "description": "Water temperature"}
        if "velocity" in variables:
            u, v, = functions.rotate_velocity(
                ds.U1.isel(MC=xi, N=yi).transpose('KMAXOUT_RESTR', 'dim_0').values,
                ds.V1.isel(M=xi, NC=yi).transpose('KMAXOUT_RESTR', 'dim_0').values,
                ds.ALFAS.isel(M=xi, N=yi).values[np.newaxis, :])
            output["variables"]["u"] = {"data": functions.filter_parameter(u, decimals=5), "unit": "m/s",
                                        "description": "Eastward flow velocity"}
            output["variables"]["v"] = {"data": functions.filter_parameter(u, decimals=5), "unit": "m/s",
                                        "description": "Northward flow velocity"}
    return output


def get_simulations_transect_period(filesystem, model, lake, start, end, latitude_list, longitude_list, variables):
    if model == "delft3d-flow":
        return get_simulations_transect_period_delft3dflow(filesystem, lake, start, end, latitude_list, longitude_list, variables)
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

    for i, file in enumerate(files):
        if not os.path.isfile(file):
            raise HTTPException(status_code=400,
                                detail="Apologies data is not available for {} week starting {}".format(lake, weeks[i]))

    start_datetime = datetime.strptime(start[0:10], "%Y%m%d%H").replace(tzinfo=timezone.utc)
    end_datetime = datetime.strptime(end[0:10], "%Y%m%d%H").replace(tzinfo=timezone.utc)

    with xr.open_mfdataset(files) as ds:
        ds['time'] = ds.indexes['time'].tz_localize('UTC')
        ds = ds.sel(time=slice(start_datetime, end_datetime))
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
        output = {"time": functions.alplakes_time(ds.time[:].values, "nano").tolist(),
                  "distance": {"data": functions.filter_parameter(sp_arr), "unit": "m",
                               "description": "Distance along transect"},
                  "depth": {"data": functions.filter_parameter(depth), "unit": "m",
                           "description": "Distance from the surface"},
                  "lat": functions.filter_parameter(lat_arr, decimals=5),
                  "lng": functions.filter_parameter(lng_arr, decimals=5),
                  "variables": {}
                  }
        if "temperature" in variables:
            t = ds.R1.isel(M=xi, N=yi, LSTSCI=0).transpose('time', 'KMAXOUT_RESTR', 'dim_0').values
            output["variables"]["temperature"] = {"data": functions.filter_parameter(t), "unit": "degC", "description": "Water temperature"}
        if "velocity" in variables:
            if "time" in ds.ALFAS.dims:
                alfas = ds.ALFAS.isel(M=xi, N=yi).transpose('time', 'dim_0').values[:, np.newaxis, :]
            else:
                alfas = ds.ALFAS.isel(M=xi, N=yi).values
            u, v, = functions.rotate_velocity(
                ds.U1.isel(MC=xi, N=yi).transpose('time', 'KMAXOUT_RESTR', 'dim_0').values,
                ds.V1.isel(M=xi, NC=yi).transpose('time', 'KMAXOUT_RESTR', 'dim_0').values,
                alfas)
            output["variables"]["u"] = {"data": functions.filter_parameter(u, decimals=5), "unit": "m/s", "description": "Eastward flow velocity"}
            output["variables"]["v"] = {"data": functions.filter_parameter(u, decimals=5), "unit": "m/s", "description": "Northward flow velocity"}
    return output


class OneDimensionalModels(str, Enum):
    simstrat = "simstrat"


class SimstratResampleOptions(str, Enum):
    daily = "daily"
    monthly = "monthly"
    yearly = "yearly"


def get_one_dimensional_metadata(filesystem):
    metadata = []
    models = os.listdir(os.path.join(filesystem, "media/1dsimulations"))

    for model in models:
        lakes = os.listdir(os.path.join(filesystem, "media/1dsimulations", model, "results"))
        m = {"model": model, "lakes": []}

        for lake in lakes:
            try:
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

                    for d in functions.monthrange(start_date, end_date, months=1):
                        if d.strftime('%Y%m') not in combined:
                            missing_dates.append(d.strftime("%Y-%m"))

                    m["lakes"].append({"name": lake,
                                       "depths": depths,
                                       "start_date": start_date.strftime("%Y-%m-%d"),
                                       "end_date": end_date.strftime("%Y-%m-%d"),
                                       "missing_dates": missing_dates})
                else:
                    raise ValueError("Model not recognised.")
            except:
                m["lakes"].append({"name": lake,
                                   "depths": [],
                                   "start_date": "NA",
                                   "end_date": "NA",
                                   "missing_dates": []})
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
            out["depths"] = depths

            variables = []
            for var in nc.variables:
                long_name = nc.variables[var].long_name if 'long_name' in nc.variables[var].ncattrs() else var
                variables.append({
                    "key": var,
                    "name": long_name,
                    "unit": nc.variables[var].units
                })
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


def get_one_dimensional_point(filesystem, model, lake, parameter, start_time, end_time, depth, resample=None):
    if model == "simstrat":
        return get_one_dimensional_point_simstrat(filesystem, lake, parameter, start_time, end_time, depth, resample)
    else:
        raise HTTPException(status_code=400,
                            detail="Apologies not available for {}".format(model))


def get_one_dimensional_point_simstrat(filesystem, lake, parameter, start, end, depth, resample):
    model = "simstrat"
    out = {}
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

    with xr.open_mfdataset(files) as ds:
        if parameter not in ds.variables:
            raise HTTPException(status_code=400, detail="Parameter {} is not available".format(parameter))
        ds = ds.sel(time=slice(start_datetime, end_datetime))
        if len(ds[parameter].shape) == 2:
            depths = ds.depth[:].values * - 1
            index = functions.get_closest_index(depth, depths)
            out["depth"] = {"data": depths[index], "unit": "m", "description": "Distance from the surface to the closest grid point to requested depth"}
            ds = ds.sel(depth=depths[index] * -1)

        df = pd.DataFrame({'time': pd.to_datetime(ds['time'].values).tz_localize('UTC'), 'values': ds[parameter][:].values})
        resample_options = {"hourly": "H", "daily": "D", "monthly": "M", "yearly": "Y"}
        if resample is not None:
            df.set_index('time', inplace=True)
            df = df.resample(resample_options[resample], label='left').mean(numeric_only=True)
            df = df.reset_index()
            out["resample"] = resample + " mean"
        out["time"] = [x.replace(tzinfo=timezone.utc).isoformat() for x in df["time"].tolist()]
        out[parameter] = {"data": functions.filter_parameter(df["values"]), "unit": ds[parameter].units, "description": ds[parameter].long_name}
        return out


def get_one_dimensional_depth_time(filesystem, model, lake, parameter, start_time, end_time):
    if model == "simstrat":
        return get_one_dimensional_depth_time_simstrat(filesystem, lake, parameter, start_time, end_time)
    else:
        raise HTTPException(status_code=400, detail="Apologies not available for {}".format(model))


def get_one_dimensional_depth_time_simstrat(filesystem, lake, parameter, start, end):
    model = "simstrat"
    out = {}
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
    with xr.open_mfdataset(files) as ds:
        if parameter not in ds.variables:
            raise HTTPException(status_code=400, detail="Parameter {} is not available".format(parameter))
        ds['time'] = ds.indexes['time'].tz_localize('UTC')
        ds = ds.sel(time=slice(start_datetime, end_datetime))
        if len(ds[parameter].shape) == 1:
            raise HTTPException(status_code=400, detail="Parameter {} exists but is not 2D".format(parameter))
        depths = ds.depth[:].values * - 1
        out["time"] = [x.replace(tzinfo=timezone.utc).isoformat() for x in
                       functions.default_time(ds.time[:].values, "nano").tolist()]
        out["depths"] = {"data": functions.filter_parameter(depths), "unit": "m", "description": "Distance from the surface to the closest grid point to requested depth"}
        out[parameter] = {"data": functions.filter_parameter(ds[parameter][:].values), "unit": ds[parameter].units, "description": ds[parameter].long_name}
        return out


def get_one_dimensional_day_of_year(filesystem, model, lake, parameter, depth):
    if model == "simstrat":
        return get_one_dimensional_day_of_year_simstrat(filesystem, lake, parameter, depth)
    else:
        raise HTTPException(status_code=400, detail="Apologies not available for {}".format(model))


def write_one_dimensional_day_of_year(filesystem, model, lake, parameter, depth):
    if model == "simstrat":
        return write_one_dimensional_day_of_year_simstrat(filesystem, lake, parameter, depth)
    else:
        raise HTTPException(status_code=400, detail="Apologies not available for {}".format(model))


def get_one_dimensional_day_of_year_simstrat(filesystem, lake, parameter, depth):
    model = "simstrat"
    lakes = os.path.join(filesystem, "media/1dsimulations", model, "results")
    if not os.path.isdir(os.path.join(lakes, lake)):
        raise HTTPException(status_code=400,
                            detail="{} simulation results are not available for {} please select from: [{}]"
                            .format(model, lake, ", ".join(os.listdir(lakes))))
    doy_file = os.path.join(filesystem, "media/1dsimulations", model, "doy", "{}_{}_{}.json".format(lake, parameter, depth))
    print(doy_file)
    if os.path.isfile(doy_file):
        with open(doy_file, "r") as f:
            out = json.load(f)
        return out
    else:
        raise HTTPException(status_code=400, detail="Apologies DOY has not been computed for your request.")


def write_one_dimensional_day_of_year_simstrat(filesystem, lake, parameter, depth):
    model = "simstrat"
    out = {}
    lakes = os.path.join(filesystem, "media/1dsimulations", model, "results")
    if not os.path.isdir(os.path.join(lakes, lake)):
        raise HTTPException(status_code=400,
                            detail="{} simulation results are not available for {} please select from: [{}]"
                            .format(model, lake, ", ".join(os.listdir(lakes))))
    doy_file = os.path.join(filesystem, "media/1dsimulations", model, "doy",
                            "{}_{}_{}.json".format(lake, parameter, float(depth)))
    files = [os.path.join(lakes, lake, file) for file in os.listdir(os.path.join(lakes, lake)) if file.endswith(".nc")]
    files.sort()
    if len(files) > 24:
        files = files[24:]  # Remove first two years as a warmup
    try:
        for i, file in enumerate(files):
            with netCDF4.Dataset(file) as nc:
                if i == 0:
                    index = functions.get_closest_index(depth, np.array(nc.variables["depth"][:]))
                    df = pd.DataFrame({'time': nc.variables["time"][:], 'value': nc.variables[parameter][index, :]})
                else:
                    df_new = pd.DataFrame({'time': nc.variables["time"][:], 'value': nc.variables[parameter][index, :]})
                    df = pd.concat([df, df_new])
        last_year = pd.Timestamp.now().year - 1
        df["time"] = pd.to_datetime(df['time'], unit='s', utc=True)
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
        out["min_date"] = df.index.min().isoformat()
        out["max_date"] = df.index.max().isoformat()
        out["doy"] = list(range(1, 367))
        out["mean"] = functions.filter_parameter(mean_values_doy.values)
        out["max"] = functions.filter_parameter(max_values_doy.values)
        out["min"] = functions.filter_parameter(min_values_doy.values)
        out["std"] = functions.filter_parameter(std_values_doy.values)
        out["perc5"] = functions.filter_parameter(percentile_5_doy.values)
        out["perc25"] = functions.filter_parameter(percentile_25_doy.values)
        out["perc75"] = functions.filter_parameter(percentile_75_doy.values)
        out["perc95"] = functions.filter_parameter(percentile_95_doy.values)
        out["lastyear"] = functions.filter_parameter(daily_average_last_year.values)

        os.makedirs(os.path.dirname(doy_file), exist_ok=True)
        with open(doy_file, "w") as f:
            json.dump(out, f)
        print("Succeeded in producing DOY for {}".format(lake))
    except Exception as e:
        print(e)
        print("Failed to produce DOY for {}".format(lake))
