import os
import json
import netCDF4
import numpy as np
import xarray as xr
import pandas as pd
from enum import Enum
from fastapi import HTTPException
from fastapi.responses import FileResponse
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
                                       "start_date": start_date.strftime("%Y-%m-%d %H:%M"),
                                       "end_date": end_date.strftime("%Y-%m-%d %H:%M"),
                                       "missing_dates": missing_dates,
                                       "height": height,
                                       "width": width})
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
                "start_date": start_date.strftime("%Y-%m-%d %H:%M"),
                "end_date": end_date.strftime("%Y-%m-%d %H:%M"),
                "missing_weeks": missing_dates,
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


def get_simulations_file(filesystem, model, lake, sunday):
    path = os.path.join(filesystem, "media/simulations", model, "results", lake, "{}.nc".format(sunday))
    if os.path.isfile(path):
        return FileResponse(path, media_type="application/nc", filename="{}_{}_{}.nc".format(model, lake, sunday))
    else:
        raise HTTPException(status_code=400,
                            detail="Apologies data is not available for {} on the week beginning {}".format(model,
                                                                                                            sunday))


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

    if len(latitude_list) != len(longitude_list):
        raise HTTPException(status_code=400,
                            detail="Latitude list and longitude list are not the same length.")

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
        grid_spacing = functions.average_grid_spacing(lat_grid, lng_grid)

        start = 0
        xi_arr, yi_arr, sp_arr, vd_arr = np.array([]), np.array([]), np.array([]), np.array([])
        for i in range(len(latitude_list) - 1):
            xi, yi, sp, vd, distance = functions.exact_line_segments(latitude_list[i], longitude_list[i],
                                                                     latitude_list[i + 1], longitude_list[i + 1],
                                                                     lat_grid, lng_grid, start, grid_spacing)
            start = start + distance
            xi_arr = np.concatenate((xi_arr, xi), axis=0)
            yi_arr = np.concatenate((yi_arr, yi), axis=0)
            sp_arr = np.concatenate((sp_arr, sp), axis=0)
            vd_arr = np.concatenate((vd_arr, vd), axis=0)

        xi_arr = xi_arr.astype(int)
        yi_arr = yi_arr.astype(int)

        lat_arr = lat_grid[xi_arr, yi_arr]
        lng_arr = lng_grid[xi_arr, yi_arr]

        idx = np.where(vd_arr == 0)[0]
        t = np.array(nc.variables["R1"][time_index, 0, :, :, :])
        t = t[:, xi_arr, yi_arr]
        t[:, idx] = -999.
        u, v, = functions.rotate_velocity(nc.variables["U1"][time_index, :, :, :],
                                          nc.variables["V1"][time_index, :, :, :],
                                          nc.variables["ALFAS"][xi_arr[i], yi_arr[i]])
        u = u[:, xi_arr, yi_arr]
        v = v[:, xi_arr, yi_arr]
        u[:, idx] = -999.
        v[:, idx] = -999.

        index = 0
        for i in range(t.shape[0]):
            if not np.all(t[i] == nodata):
                index = i
                break

        depth = depth[index:]
        t = t[index:, :]
        u = u[index:, :]
        v = v[index:, :]

        output = {"lake": lake,
                  "datetime": functions.convert_from_unit(time[time_index], nc.variables["time"].units),
                  "distance": functions.filter_parameter(sp_arr),
                  "latitude": functions.filter_parameter(lat_arr, decimals=5),
                  "longitude": functions.filter_parameter(lng_arr, decimals=5),
                  "depth": {"data": functions.filter_parameter(depth), "unit": "m"},
                  "temperature": {"data": functions.filter_parameter(t), "unit": "degC"},
                  "u": {"data": functions.filter_parameter(u, decimals=5), "unit": nc.variables["U1"].units},
                  "v": {"data": functions.filter_parameter(v, decimals=5), "unit": nc.variables["V1"].units}}
    return output


def get_simulations_transect_period(filesystem, model, lake, start, end, latitude_list, longitude_list):
    if model == "delft3d-flow":
        return get_simulations_transect_period_delft3dflow(filesystem, lake, start, end, latitude_list, longitude_list)
    else:
        raise HTTPException(status_code=400,
                            detail="Apologies profile extraction not available for {}".format(model))


def get_simulations_transect_period_delft3dflow(filesystem, lake, start, end, latitude_str, longitude_str,
                                                nodata=-999.0, velocity=False):
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

        output = {"lake": lake,
                  "distance": functions.filter_parameter(sp_arr),
                  "latitude": functions.filter_parameter(lat_arr, decimals=5),
                  "longitude": functions.filter_parameter(lng_arr, decimals=5),
                  }

        idx = np.where(vd_arr == 0)[0]

        t_s = ds.R1.isel(M=xr.DataArray(xi_arr), N=xr.DataArray(yi_arr))
        t = t_s[:, 0, :].values
        index = 0
        for i in range(t.shape[1]):
            if not np.all(t[0, i] == nodata):
                index = i
                break
        depth = z[index:]
        t = t[:, index:, :]
        t[:, :, idx] = -999.

        if velocity:
            u_s = ds.U1.isel(MC=xr.DataArray(xi_arr), N=xr.DataArray(yi_arr))
            v_s = ds.V1.isel(M=xr.DataArray(xi_arr), NC=xr.DataArray(yi_arr))
            a_s = ds.ALFAS.isel(M=xr.DataArray(xi_arr), N=xr.DataArray(yi_arr))
            a_e = a_s[:].values[:, np.newaxis, :]
            u, v, = functions.rotate_velocity(u_s[:, index:, :].values, v_s[:, index:, :].values, a_e)
            u[:, :, idx] = -999.
            v[:, :, idx] = -999.
            output["u"] = {"data": functions.filter_parameter(u, decimals=5), "unit": "m/s"}
            output["v"] = {"data": functions.filter_parameter(u, decimals=5), "unit": "m/s"}

    output["time"] = functions.alplakes_time(ds.time[:].values, "nano").tolist()
    output["depth"] = {"data": functions.filter_parameter(depth), "unit": "m"}
    output["temperature"] = {"data": functions.filter_parameter(t), "unit": "degC"}

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


def get_simulations_point(filesystem, model, lake, start, end, depth, latitude, longitude):
    if model == "delft3d-flow":
        return get_simulations_point_delft3dflow(filesystem, lake, start, end, depth, latitude, longitude)
    else:
        raise HTTPException(status_code=400,
                            detail="Apologies profile extraction not available for {}".format(model))


def get_simulations_point_delft3dflow(filesystem, lake, start, end, depth, latitude, longitude, nodata=-999.0):
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

            depth_index = functions.get_closest_index(depth, np.array(nc.variables["ZK_LYR"][:]) * -1)
            depth = nc.variables["ZK_LYR"][depth_index].tolist() * -1

            lat_grid, lng_grid = functions.coordinates_to_latlng(nc.variables["XZ"][:], nc.variables["YZ"][:])
            x_index, y_index, distance = functions.get_closest_location(latitude, longitude, lat_grid, lng_grid)

            t = np.array(nc.variables["R1"][time_index_start:time_index_end, 0, depth_index, x_index, y_index])
            u, v, = functions.rotate_velocity(
                nc.variables["U1"][time_index_start:time_index_end, depth_index, x_index, y_index],
                nc.variables["V1"][time_index_start:time_index_end, depth_index, x_index, y_index],
                nc.variables["ALFAS"][x_index, y_index])
            at = functions.alplakes_time(time[time_index_start:time_index_end], nc.variables["time"].units)

            if len(t_out) == 0:
                t_out, u_out, v_out, at_out = t, u, v, at
            else:
                t_out = np.concatenate((t_out, t), axis=0)
                u_out = np.concatenate((u_out, u), axis=0)
                v_out = np.concatenate((v_out, v), axis=0)
                at_out = np.concatenate((at_out, at), axis=0)

    output = {"lake": lake,
              "time": at_out.tolist(),
              "latitude": lat_grid[x_index, y_index],
              "longitude": lng_grid[x_index, y_index],
              "distance": distance,
              "depth": {"value": depth, "unit": "m"},
              "temperature": {"data": functions.filter_parameter(t_out), "unit": "degC"},
              "u": {"data": functions.filter_parameter(u_out, decimals=5), "unit": "m/s"},
              "v": {"data": functions.filter_parameter(v_out, decimals=5), "unit": "m/s"}}

    return output


def get_simulations_layer_average_temperature(filesystem, model, lake, start, end, depth):
    if model == "delft3d-flow":
        return get_simulations_layer_average_temperature_delft3dflow(filesystem, lake, start, end, depth)
    else:
        raise HTTPException(status_code=400,
                            detail="Apologies data is not available for {}".format(model))


def get_simulations_layer_average_temperature_delft3dflow(filesystem, lake, start, end, depth):
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
            p = functions.alplakes_parameter(
                nc.variables["R1"][time_index_start:time_index_end, 0, depth_index, :])
            p = np.nanmean(p, axis=(1, 2))
            t = functions.unix_time(time[time_index_start:time_index_end], nc.variables["time"].units)
            if out is None:
                out = p
                times = t
            else:
                out = np.concatenate((out, p), axis=0)
                times = np.concatenate((times, t), axis=0)
    output = {"date": functions.filter_parameter(times), "temperature": functions.filter_parameter(out)}
    return output


class OneDimensionalModels(str, Enum):
    simstrat = "simstrat"


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
                                       "start_date": start_date.strftime("%Y-%m-%d %H:%M"),
                                       "end_date": end_date.strftime("%Y-%m-%d %H:%M"),
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
            out["start_date"] = start_date.strftime("%Y-%m-%d %H:%M")

        with netCDF4.Dataset(os.path.join(path, files[-1])) as nc:
            end_date = functions.convert_from_unit(nc.variables["time"][-1], nc.variables["time"].units)
            out["end_date"] = end_date.strftime("%Y-%m-%d %H:%M")
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


def get_one_dimensional_point(filesystem, model, lake, parameter, start_time, end_time, depth):
    if model == "simstrat":
        return get_one_dimensional_point_simstrat(filesystem, lake, parameter, start_time, end_time, depth)
    else:
        raise HTTPException(status_code=400,
                            detail="Apologies not available for {}".format(model))


def get_one_dimensional_point_simstrat(filesystem, lake, parameter, start, end, depth):
    model = "simstrat"
    out = {"lake": lake, "model": model}
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

    start_datetime = datetime.strptime(start, "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
    end_datetime = datetime.strptime(end, "%Y%m%d%H%M").replace(tzinfo=timezone.utc)

    with xr.open_mfdataset(files) as ds:
        if parameter not in ds.variables:
            raise HTTPException(status_code=400, detail="Parameter {} is not available".format(parameter))
        ds['time'] = ds.indexes['time'].tz_localize('UTC')
        ds = ds.sel(time=slice(start_datetime, end_datetime))

        if len(ds[parameter].shape) == 2:
            depths = ds.depth[:].values * - 1
            index = functions.get_closest_index(depth, depths)
            out["depth"] = depths[index]
            values = ds[parameter][index, :].values
        else:
            values = ds[parameter][:].values
        out["time"] = functions.default_time(ds.time[:].values, "nano").tolist()
        out[parameter] = functions.filter_parameter(values)
        out["unit"] = ds[parameter].units
        return out


def get_one_dimensional_depth_time(filesystem, model, lake, parameter, start_time, end_time):
    if model == "simstrat":
        return get_one_dimensional_depth_time_simstrat(filesystem, lake, parameter, start_time, end_time)
    else:
        raise HTTPException(status_code=400, detail="Apologies not available for {}".format(model))


def get_one_dimensional_depth_time_simstrat(filesystem, lake, parameter, start, end):
    model = "simstrat"
    out = {"lake": lake, "model": model}
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
        out["depths"] = functions.filter_parameter(depths)
        out["time"] = functions.default_time(ds.time[:].values, "nano").tolist()
        out[parameter] = functions.filter_parameter(ds[parameter][:].values)
        out["unit"] = ds[parameter].units
        return out


def get_one_dimensional_day_of_year(filesystem, model, lake, parameter, depth):
    if model == "simstrat":
        return get_one_dimensional_day_of_year_simstrat(filesystem, lake, parameter, depth)
    else:
        raise HTTPException(status_code=400, detail="Apologies not available for {}".format(model))


def get_one_dimensional_day_of_year_simstrat(filesystem, lake, parameter, depth):
    model = "simstrat"
    out = {"lake": lake, "model": model}
    lakes = os.path.join(filesystem, "media/1dsimulations", model, "results")
    if not os.path.isdir(os.path.join(lakes, lake)):
        raise HTTPException(status_code=400,
                            detail="{} simulation results are not available for {} please select from: [{}]"
                            .format(model, lake, ", ".join(os.listdir(lakes))))
    doy_file = os.path.join(filesystem, "media/1dsimulations", model, "doy", "{}_{}.json".format(lake, depth))
    if os.path.isfile(doy_file):
        with open(doy_file, "r") as f:
            out = json.load(f)
        return out

    files = [os.path.join(lakes, lake, file) for file in os.listdir(os.path.join(lakes, lake)) if file.endswith(".nc")]
    files.sort()
    files = files[24:]  # Remove first two years as a warmup
    with xr.open_mfdataset(files) as ds:
        if parameter not in ds.variables:
            raise HTTPException(status_code=400, detail="Parameter {} is not available".format(parameter))
        ds['time'] = ds.indexes['time'].tz_localize('UTC')
        ds['time'] = pd.to_datetime(ds['time'].values)
        ds = ds.sel(time=ds['time.year'] != pd.Timestamp.now().year)
        out["unit"] = ds[parameter].units
        if len(ds[parameter].shape) == 2:
            depths = ds.depth[:].values * - 1
            index = functions.get_closest_index(depth, depths)
            out["depth"] = depths[index]
            data = ds[parameter].isel({"depth": index})
        else:
            data = ds[parameter]

        max_values_doy = data.groupby('time.dayofyear').max(dim='time')
        mean_values_doy = data.groupby('time.dayofyear').mean(dim='time')
        min_values_doy = data.groupby('time.dayofyear').min(dim='time')
        std_values_doy = data.groupby('time.dayofyear').std(dim='time')

        out["doy"] = list(range(1, 367))
        out["mean"] = functions.filter_parameter(mean_values_doy.values)
        out["max"] = functions.filter_parameter(max_values_doy.values)
        out["min"] = functions.filter_parameter(min_values_doy.values)
        out["std"] = functions.filter_parameter(std_values_doy.values)
    os.makedirs(os.path.dirname(doy_file), exist_ok=True)
    with open(doy_file, "w") as f:
        json.dump(out, f)
    return out
