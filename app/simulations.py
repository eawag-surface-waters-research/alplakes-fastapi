import os
import shutil
import netCDF4
import numpy as np
from enum import Enum
from pydantic import BaseModel
from fastapi import HTTPException
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta, SU

from app import functions


class Models(str, Enum):
    delft3dflow = "delft3d-flow"


class Lakes(str, Enum):
    geneva = "geneva"
    greifensee = "greifensee"
    zurich = "zurich"
    biel = "biel"


def verify_simulations_layer(model, lake, datetime, depth):
    return True


def get_simulations_layer(filesystem, model, lake, time, depth):
    if model == "delft3d-flow":
        get_simulations_layer_delft3dflow(filesystem, lake, time, depth)
    else:
        raise HTTPException(status_code=400,
                            detail="Apologies data is not available for {}".format(model))


def get_simulations_layer_delft3dflow(filesystem, lake, time, depth):
    model = "delft3d-flow"
    origin = datetime.strptime(time, "%Y%m%d%H%M")
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
        depth_index = functions.get_closest_index(depth, np.array(nc.variables["depth"][:]) * -1)

        return {"time": {"name": nc.variables["time"].long_name,
                         "units": nc.variables["time"].units,
                         "data": nc.variables["time"][time_index]},
                "depth": {"name": nc.variables["depth"].long_name,
                          "units": nc.variables["depth"].units,
                          "data": nc.variables["depth"][depth_index]},
                "x": {"name": nc.variables["x"].long_name,
                      "units": nc.variables["x"].units,
                      "data": nc.variables["x"][:]},
                "y": {"name": nc.variables["y"].long_name,
                      "units": nc.variables["y"].units,
                      "data": nc.variables["y"][:]},
                "t": {"name": nc.variables["t"].long_name,
                      "units": nc.variables["t"].units,
                      "data": nc.variables["t"][time_index, depth_index, :]},
                "u": {"name": nc.variables["u"].long_name,
                      "units": nc.variables["u"].units,
                      "data": nc.variables["u"][time_index, depth_index, :]},
                "v": {"name": nc.variables["v"].long_name,
                      "units": nc.variables["v"].units,
                      "data": nc.variables["v"][time_index, depth_index, :]},
                }


class Notification(BaseModel):
    type: str
    model: str
    value: str


def notify_new_delft3dflow(filesystem, model, value):
    lake = value.split("_")[-2]
    file = value.split("/")[-1]
    folder = os.path.join(filesystem, "media/simulations", model, "results", lake)
    local = os.path.join(folder, file)

    functions.download_file(value, local)

    with netCDF4.Dataset(local, "r") as nc:
        time = np.array(nc.variables["time"][:])
        time_unit = nc.variables["time"].units
        min_time = functions.convert_from_unit(np.min(time), time_unit)
        max_time = functions.convert_from_unit(np.max(time), time_unit)
        start_time = min_time + relativedelta(weekday=SU(-1))
        end_time = start_time + timedelta(days=7)
        while start_time < max_time:
            idx = np.where(np.logical_and(time >= functions.convert_to_unit(start_time, time_unit),
                                          time < functions.convert_to_unit(end_time, time_unit)))
            s = np.min(idx)
            e = np.max(idx) + 1
            temp_file_name = os.path.join(folder, "temp_{}.nc".format(start_time.strftime('%Y%m%d')))
            final_file_name = os.path.join(folder, "{}.nc".format(start_time.strftime('%Y%m%d')))
            if start_time != min_time + relativedelta(weekday=SU(-1)) and os.path.isfile(final_file_name):
                with netCDF4.Dataset(final_file_name, "r") as temp:
                    if len(temp.variables["time"][:]) >= len(idx):
                        start_time = start_time + timedelta(days=7)
                        end_time = end_time + timedelta(days=7)
                        continue

            with netCDF4.Dataset(temp_file_name, "w") as dst:
                # Copy Attributes
                dst.setncatts(nc.__dict__)
                # Copy Dimensions
                for name, dimension in nc.dimensions.items():
                    dst.createDimension(name, (len(dimension) if not dimension.isunlimited() else None))
                # Copy Variables
                for name, variable in nc.variables.items():
                    x = dst.createVariable(name, variable.datatype, variable.dimensions)
                    if "time" in list(variable.dimensions):
                        if list(variable.dimensions)[0] != "time":
                            raise ValueError("Code only works with time as first dimension.")
                        if len(variable.dimensions) > 1:
                            dst[name][:] = nc[name][s:e, :]
                        else:
                            dst[name][:] = nc[name][s:e]
                    else:
                        dst[name][:] = nc[name][:]
                    dst[name].setncatts(nc[name].__dict__)
            shutil.move(temp_file_name, final_file_name)
            start_time = start_time + timedelta(days=7)
            end_time = end_time + timedelta(days=7)

    os.remove(local)
