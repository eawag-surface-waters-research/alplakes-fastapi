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
            try:
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
            except:
                m["lakes"].append({"name": lake,
                                   "depths": [],
                                   "start_date": "NA",
                                   "end_date": "NA",
                                   "missing_dates": []})
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


def verify_simulations_profile(model, lake, dt, latitude, longitude):
    return True


def verify_simulations_transect(model, lake, dt, latitude_list, longitude_list):
    return True


def verify_simulations_depthtime(model, lake, start, end, latitude, longitude):
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
        x = functions.filter_parameter(nc.variables["XZ"][:], nodata=0.0)
        y = functions.filter_parameter(nc.variables["YZ"][:], nodata=0.0)
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
                     "data": functions.filter_parameter(u, decimals=5)},
               "v": {"name": "Water velocity (East)",
                     "units": "m/s",
                     "data": functions.filter_parameter(v, decimals=5)}
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
                lat_grid, lng_grid = functions.coordinates_to_latlng(nc.variables["XZ"][:], nc.variables["YZ"][:])
                geometry = np.concatenate((lat_grid, lng_grid), axis=1)
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
        string_arr += (times[timestep] + "\n" + '\n'.join(','.join(f % x for x in y) for y in out[timestep, :]).replace(
            "nan", "") + "\n")

    return string_arr


def get_simulations_profile(filesystem, model, lake, dt, latitude, longitude):
    if model == "delft3d-flow":
        return get_simulations_profile_delft3dflow(filesystem, lake, dt, latitude, longitude)
    else:
        raise HTTPException(status_code=400,
                            detail="Apologies profile extraction not available for {}".format(model))


def get_simulations_profile_delft3dflow(filesystem, lake, dt, latitude, longitude):
    model = "delft3d-flow"
    lakes = os.path.join(filesystem, "media/simulations", model, "results")
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

        output = {"lake": lake,
                  "datetime": functions.convert_from_unit(time[time_index], nc.variables["time"].units),
                  "latitude": lat_grid[x_index, y_index],
                  "longitude": lng_grid[x_index, y_index],
                  "distance": distance,
                  "depth": {"data": functions.filter_parameter(depth), "unit": "m"},
                  "temperature": {"data": t, "unit": "degC"},
                  "u": {"data": functions.filter_parameter(u, decimals=5), "unit": nc.variables["U1"].units},
                  "v": {"data": functions.filter_parameter(v, decimals=5), "unit": nc.variables["V1"].units}}
    return output


def get_simulations_transect(filesystem, model, lake, dt, latitude_list, longitude_list):
    if model == "delft3d-flow":
        return get_simulations_transect_delft3dflow(filesystem, lake, dt, latitude_list, longitude_list)
    else:
        raise HTTPException(status_code=400,
                            detail="Apologies profile extraction not available for {}".format(model))


def get_simulations_transect_delft3dflow(filesystem, lake, dt, latitude_str, longitude_str, nodata=-999.0):
    model = "delft3d-flow"

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

        start = 0
        xi_arr, yi_arr, sp_arr = np.array([]), np.array([]), np.array([])
        for i in range(len(latitude_list) - 1):
            xi, yi, sp, distance = functions.exact_line_segments(latitude_list[i], longitude_list[i],
                                                                 latitude_list[i + 1],
                                                                 longitude_list[i + 1], lat_grid, lng_grid, start, 2000)
            start = start + distance
            xi_arr = np.concatenate((xi_arr, xi), axis=0)
            yi_arr = np.concatenate((yi_arr, yi), axis=0)
            sp_arr = np.concatenate((sp_arr, sp), axis=0)

        xi_arr = xi_arr.astype(int)
        yi_arr = yi_arr.astype(int)

        lat_arr = lat_grid[xi_arr, yi_arr]
        lng_arr = lng_grid[xi_arr, yi_arr]

        t_arr = []
        for i in range(len(xi)):
            t = np.array(nc.variables["R1"][time_index, 0, :, xi[i], yi[i]])
            u, v, = functions.rotate_velocity(nc.variables["U1"][time_index, :, xi[i], yi[i]],
                                              nc.variables["V1"][time_index, :, xi[i], yi[i]],
                                              nc.variables["ALFAS"][xi[i], yi[i]])
            if len(t_arr) == 0:
                t_arr, u_arr, v_arr = t.reshape(-1, 1), u.reshape(-1, 1), v.reshape(-1, 1)
            else:
                t_arr = np.concatenate((t_arr, t.reshape(-1, 1)), axis=1)
                u_arr = np.concatenate((u_arr, u.reshape(-1, 1)), axis=1)
                v_arr = np.concatenate((v_arr, v.reshape(-1, 1)), axis=1)

        index = 0
        for i in range(t_arr.shape[0]):
            if not np.all(t_arr[i] == nodata):
                index = i
                break

        depth = depth[index:]
        t_arr = t_arr[index:, :]
        u_arr = u_arr[index:, :]
        v_arr = v_arr[index:, :]

        output = {"lake": lake,
                  "datetime": functions.convert_from_unit(time[time_index], nc.variables["time"].units),
                  "distance": functions.filter_parameter(sp_arr),
                  "latitude": functions.filter_parameter(lat_arr, decimals=5),
                  "longitude": functions.filter_parameter(lng_arr, decimals=5),
                  "depth": {"data": functions.filter_parameter(depth), "unit": "m"},
                  "temperature": {"data": functions.filter_parameter(t_arr), "unit": "degC"},
                  "u": {"data": functions.filter_parameter(u_arr, decimals=5), "unit": nc.variables["U1"].units},
                  "v": {"data": functions.filter_parameter(v_arr, decimals=5), "unit": nc.variables["V1"].units}}
    return output


def get_simulations_depthtime(filesystem, model, lake, start, end, latitude, longitude):
    if model == "delft3d-flow":
        return get_simulations_depthtime_delft3dflow(filesystem, lake, start, end, latitude, longitude)
    else:
        raise HTTPException(status_code=400,
                            detail="Apologies profile extraction not available for {}".format(model))


def get_simulations_depthtime_delft3dflow(filesystem, lake, start, end, latitude, longitude, nodata=-999.0):
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

    t_out = []
    for week in weeks:
        with netCDF4.Dataset(os.path.join(lakes, lake, "{}.nc".format(week.strftime("%Y%m%d")))) as nc:
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

            depth = (np.array(nc.variables["ZK_LYR"][:]) * -1).tolist()
            lat_grid, lng_grid = functions.coordinates_to_latlng(nc.variables["XZ"][:], nc.variables["YZ"][:])
            x_index, y_index, distance = functions.get_closest_location(latitude, longitude, lat_grid, lng_grid)

            t = np.array(nc.variables["R1"][time_index_start:time_index_end, 0, :, x_index, y_index])
            u, v, = functions.rotate_velocity(nc.variables["U1"][time_index_start:time_index_end, :, x_index, y_index],
                                              nc.variables["V1"][time_index_start:time_index_end, :, x_index, y_index],
                                              nc.variables["ALFAS"][x_index, y_index])
            at = functions.alplakes_time(time[time_index_start:time_index_end], nc.variables["time"].units)

            if len(t_out) == 0:
                t_out, u_out, v_out, at_out = t, u, v, at
            else:
                t_out = np.concatenate((t_out, t), axis=0)
                u_out = np.concatenate((u_out, u), axis=0)
                v_out = np.concatenate((v_out, v), axis=0)
                at_out = np.concatenate((at_out, at), axis=0)

    t_out = t_out.T
    u_out = u_out.T
    v_out = v_out.T

    index = 0
    for i in range(t_out.shape[0]):
        if not np.all(t_out[i] == nodata):
            index = i
            break

    depth = depth[index:]
    t_out = t_out[index:, :]
    u_out = u_out[index:, :]
    v_out = v_out[index:, :]

    output = {"lake": lake,
              "time": at_out.tolist(),
              "latitude": lat_grid[x_index, y_index],
              "longitude": lng_grid[x_index, y_index],
              "distance": distance,
              "depth": {"data": functions.filter_parameter(depth), "unit": "m"},
              "temperature": {"data": functions.filter_parameter(t_out), "unit": "degC"},
              "u": {"data": functions.filter_parameter(u_out, decimals=5), "unit": "m/s"},
              "v": {"data": functions.filter_parameter(v_out, decimals=5), "unit": "m/s"}}

    return output
