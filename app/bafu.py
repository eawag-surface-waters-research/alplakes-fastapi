import os
import numpy as np
import pandas as pd
from enum import Enum
from datetime import datetime, timedelta
from fastapi import HTTPException
from fastapi.responses import FileResponse


def verify_hydrodata_measured(station_id, parameter, start_date, end_date):
    return True


def get_hydrodata_measured(filesystem, station_id, parameter, start_date, end_date):
    station_dir = os.path.join(filesystem, "media/bafu/hydrodata/CSV", str(station_id))
    if not os.path.exists(station_dir):
        raise HTTPException(status_code=400,
                            detail="No data available for station id: {}".format(station_id))
    if not os.path.exists(os.path.join(station_dir, parameter)):
        raise HTTPException(status_code=400,
                            detail='Parameter "{}" not available for station {}, please select from: {}'.format(parameter, station_id, ", ".join(os.listdir(station_dir))))
    folder = os.path.join(station_dir, parameter)
    start_date = datetime.strptime(start_date, '%Y%m%d')
    end_date = datetime.strptime(end_date, '%Y%m%d')
    files = [os.path.join(folder, "BAFU_{}_{}_{}.csv".format(station_id, parameter, (start_date+timedelta(days=x)).strftime("%Y-%m-%d")))
             for x in range((end_date-start_date).days + 1)]
    bad_files = []
    for file in files:
        if not os.path.isfile(file):
            bad_files.append(file.split("/")[-1].split(".")[0][-10:])
    if len(bad_files) > 0:
        raise HTTPException(status_code=400,
                            detail="Data not available for station {} ({}) for the following dates: {}".format(station_id, parameter, ", ".join(bad_files)))

    df = pd.concat(map(pd.read_csv, files), ignore_index=True)
    return {"Time": list(df["Time"]), parameter: list(df["BAFU_{}_{}".format(station_id, parameter)])}


class HydrodataPredicted(str, Enum):
    official = "official"
    unofficial = "unofficial"


def verify_hydrodata_predicted(status, station_id, parameter):
    return True


def get_hydrodata_predicted(filesystem, status, station_id, parameter):
    file = os.path.join(filesystem, "media/bafu/hydrodata", "pqprevi-" + status, "Pqprevi_{}_{}.txt".format(parameter, station_id))
    if not os.path.exists(file):
        raise HTTPException(status_code=400,
                            detail="Prediction not available for parameter {} at station {}.".format(parameter, station_id))
    return FileResponse(file)


def metadata_hydrodata_total_lake_inflow(filesystem):
    output = []
    folder = os.path.join(filesystem, "media/bafu/hydrodata/TotalInflowLakes")
    lakes = os.listdir(folder)
    for lake in lakes:
        output.append({"lake": lake, "parameters": os.listdir(os.path.join(folder, lake))})
    return output


def verify_hydrodata_total_lake_inflow(lake, parameter, start_date, end_date):
    return True


def get_hydrodata_total_lake_inflow(filesystem, lake, parameter, start_date, end_date):
    lake_dir = os.path.join(filesystem, "media/bafu/hydrodata/TotalInflowLakes", str(lake))
    if not os.path.exists(lake_dir):
        raise HTTPException(status_code=400,
                            detail="No data available for lake: {}".format(lake))
    if not os.path.exists(os.path.join(lake_dir, parameter)):
        raise HTTPException(status_code=400,
                            detail='Parameter "{}" not available for {}, please select from: {}'.format(
                                parameter, lake, ", ".join(os.listdir(lake_dir))))
    folder = os.path.join(lake_dir, parameter)
    start_date = datetime.strptime(start_date, '%Y%m%d')
    end_date = datetime.strptime(end_date, '%Y%m%d')
    files = [os.path.join(folder, "{}_{}_{}.csv".format(lake, parameter,
                                                             (start_date + timedelta(days=x)).strftime("%Y-%m-%d")))
             for x in range((end_date - start_date).days + 1)]
    bad_files = []
    for file in files:
        if not os.path.isfile(file):
            bad_files.append(file.split("/")[-1].split(".")[0][-10:])
    if len(bad_files) > 0:
        raise HTTPException(status_code=400,
                            detail="Data not available for {} ({}) for the following dates: {}".format(
                                lake, parameter, ", ".join(bad_files)))

    df = pd.concat(map(pd.read_csv, files), ignore_index=True)
    return df.to_json()
