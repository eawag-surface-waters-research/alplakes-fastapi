import os
import json
import shutil
import netCDF4
import logging
import numpy as np
from enum import Enum
from pydantic import BaseModel
from fastapi import HTTPException
from datetime import datetime, timedelta, timezone
from dateutil.relativedelta import relativedelta, SU

from app import functions


def get_metadata(filesystem):
    metadata = []
    models = os.listdir(os.path.join(filesystem, "media/simulations"))

    for model in models:
        lakes = os.listdir(os.path.join(filesystem, "media/simulations", model, "results"))
        m = {"model": model, "lakes": []}

        for lake in lakes:
            path = os.path.join(os.path.join(filesystem, "media/simulations", model, "results", lake))
            files = os.listdir(path)
            files = [file for file in files if len(file.split(".")[0]) == 8 and file.split(".")[1] == "nc"]
            files.sort()
            combined = '_'.join(files)
            missing_dates = []

            with netCDF4.Dataset(os.path.join(path, files[0])) as nc:
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
                               "start_date": start_date.strftime("%Y-%m-%d %H:%M"),
                               "end_date": end_date.strftime("%Y-%m-%d %H:%M"),
                               "missing_dates": missing_dates})
        metadata.append(m)
    return metadata


def verify_metadata_lake(model, lake):
    return True


def get_metadata_lake(filesystem, model, lake):
    path = os.path.join(os.path.join(filesystem, "media/simulations", model, "results", lake))
    files = os.listdir(path)
    files = [file for file in files if len(file.split(".")[0]) == 8 and file.split(".")[1] == "nc"]
    files.sort()
    combined = '_'.join(files)
    missing_dates = []

    with netCDF4.Dataset(os.path.join(path, files[0])) as nc:
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
            "start_date": start_date.strftime("%Y-%m-%d %H:%M"),
            "end_date": end_date.strftime("%Y-%m-%d %H:%M"),
            "missing_weeks": missing_dates}


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
    geometry = "geometry"
    temperature = "temperature"
    velocity = "velocity"


def verify_simulations_layer(model, lake, datetime, depth):
    return True


def verify_simulations_layer_alplakes(model, lake, parameter, start, end, depth):
    return True


def verify_simulations_profile(model, lake, parameter, datetime, latitude, longitude):
    return True


def verify_simulations_transect(model, lake, parameter, datetime, latitude_list, longitude_list):
    return True


def verify_simulations_depthtime(model, lake, parameter, start, end, latitude, longitude):
    return True


def get_simulations_layer(filesystem, model, lake, time, depth):
    if model == "delft3d-flow":
        return get_simulations_layer_delft3dflow(filesystem, lake, time, depth)
    else:
        raise HTTPException(status_code=400,
                            detail="Apologies data is not available for {}".format(model))


def get_simulations_layer_alplakes(filesystem, model, lake, parameter, start, end, depth):
    if model == "delft3d-flow":
        return get_simulations_layer_alplakes_delft3dflow(filesystem, lake, parameter, start, end, depth)
    else:
        raise HTTPException(status_code=400,
                            detail="Apologies data is not available for {}".format(model))


def get_simulations_layer_delft3dflow(filesystem, lake, time, depth):
    model = "delft3d-flow"
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
        x = functions.filter_coordinate(nc.variables["XZ"][:])
        y = functions.filter_coordinate(nc.variables["YZ"][:])
        t = functions.filter_parameter(nc.variables["R1"][time_index, 0, depth_index, :])
        u, v, = functions.rotate_velocity(nc.variables["U1"][time_index, depth_index, :],
                                          nc.variables["V1"][time_index, depth_index, :],
                                          nc.variables["ALFAS"][:])

        out = {"time": {"name": nc.variables["time"].long_name,
                        "units": nc.variables["time"].units,
                        "string": functions.convert_from_unit(time, nc.variables["time"].units).strftime(
                            "%Y-%m-%d %H:%M:%S"),
                        "data": time},
               "depth": {"name": nc.variables["ZK_LYR"].long_name,
                         "units": nc.variables["ZK_LYR"].units,
                         "data": depth},
               "x": {"name": nc.variables["XZ"].long_name,
                     "units": nc.variables["XZ"].units,
                     "data": x},
               "y": {"name": nc.variables["YZ"].long_name,
                     "units": nc.variables["YZ"].units,
                     "data": y},
               "t": {"name": "Water temperature",
                     "units": "degC",
                     "data": t},
               "u": {"name": "Water velocity (North)",
                     "units": "m/s",
                     "data": u},
               "v": {"name": "Water velocity (East)",
                     "units": "m/s",
                     "data": v}
               }
    return out


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

    start_datetime = datetime.strptime(start, "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
    end_datetime = datetime.strptime(end, "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
    out = None
    times = None
    for week in weeks:
        with netCDF4.Dataset(os.path.join(lakes, lake, "{}.nc".format(week.strftime("%Y%m%d")))) as nc:
            if parameter == "geometry":
                geometry = functions.alplakes_coordinates(nc.variables["XZ"][:], nc.variables["YZ"][:])
                return '\n'.join(','.join('%0.8f' % x for x in y) for y in geometry).replace("nan", "")
            time = np.array(nc.variables["time"][:])
            min_time = np.min(time)
            max_time = np.max(time)
            start_time = functions.convert_to_unit(start_datetime, nc.variables["time"].units)
            end_time = functions.convert_to_unit(end_datetime, nc.variables["time"].units)
            if min_time <= start_time <= max_time:
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
                p = functions.alplakes_temperature(
                    nc.variables["R1"][time_index_start:time_index_end, 0, depth_index, :])
            elif parameter == "velocity":
                f = '%0.5f'
                p = functions.alplakes_velocity(
                    nc.variables["U1"][time_index_start:time_index_end, depth_index, :],
                    nc.variables["V1"][time_index_start:time_index_end, depth_index, :],
                    nc.variables["ALFAS"][:])
            else:
                raise HTTPException(status_code=400,
                                    detail="Parameter {} not recognised, please select from: [geometry, temperature, "
                                           "velocity]".format(parameter))
            t = functions.alplakes_time(time[time_index_start:time_index_end], nc.variables["time"].units)
            if out is None:
                out = p
                times = t
            else:
                out = np.concatenate((out, p), axis=0)
                times = np.concatenate((times, t), axis=0)

    shape = out.shape
    string_arr = ""
    for timestep in range(shape[0]):
        string_arr += (times[timestep] + "\n" + '\n'.join(','.join(f % x for x in y) for y in out[timestep, :]).replace("nan", "") + "\n")

    return string_arr


def get_simulations_profile(filesystem, model, lake, parameter, datetime, latitude, longitude):
    if model == "delft3d-flow":
        return get_simulations_profile_delft3dflow(filesystem, lake, parameter, datetime, latitude, longitude)
    else:
        raise HTTPException(status_code=400,
                            detail="Apologies profile extraction not available for {}".format(model))


def get_simulations_profile_delft3dflow(filesystem, lake, parameter, datetime, latitude, longitude):
    output = {lake, parameter, datetime, latitude, longitude}
    return output


def get_simulations_transect(filesystem, model, lake, parameter, datetime, latitude_list, longitude_list):
    if model == "delft3d-flow":
        return get_simulations_transect_delft3dflow(filesystem, lake, parameter, datetime, latitude_list, longitude_list)
    else:
        raise HTTPException(status_code=400,
                            detail="Apologies profile extraction not available for {}".format(model))


def get_simulations_transect_delft3dflow(filesystem, lake, parameter, datetime, latitude_list, longitude_list):
    output = {lake, parameter, datetime, latitude_list, longitude_list}
    return output


def get_simulations_depthtime(filesystem, model, lake, parameter, start, end, latitude, longitude):
    if model == "delft3d-flow":
        return get_simulations_depthtime_delft3dflow(filesystem, lake, parameter, start, end, latitude, longitude)
    else:
        raise HTTPException(status_code=400,
                            detail="Apologies profile extraction not available for {}".format(model))


def get_simulations_depthtime_delft3dflow(filesystem, lake, parameter, start, end, latitude, longitude):
    output = {lake, parameter, start, end, latitude, longitude}
    return output
