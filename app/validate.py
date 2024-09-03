from fastapi import Path, HTTPException
from datetime import datetime


def path_latitude(example="46.5", description="Latitude (WGS 84)"):
    return Path(..., title="Latitude", ge=-90, le=90, description=description, example=example)


def path_longitude(example="6.67", description="Longitude (WGS 84)"):
    return Path(..., title="Longitude", ge=-180, le=180, description=description, example=example)


def path_time(description="The time in YYYYmmddHHMM format (UTC)", example="202304050300"):
    return Path(..., regex=r"^\d{12}$", title="Datetime", description=description, example=example)


def path_date(description="The date in YYYYmmdd format", example="20230101"):
    return Path(..., regex=r"^\d{8}$", title="Date", description=description, example=example)


def path_month(description="The date in YYYYmm format", example="202405"):
    return Path(..., regex=r"^\d{6}$", title="Date", description=description, example=example)


def path_depth(example="1", description="Depth (m)"):
    return Path(..., title="Depth", ge=0, le=500, description=description, example=example)


def date(value):
    year = int(value[:4])
    month = int(value[4:6])
    day = int(value[6:])
    if year < 1000 or month < 1 or month > 12 or day < 1 or day > 31:
        raise HTTPException(status_code=400, detail="Invalid date format: {}".format(value))


def time(value):
    year = int(value[:4])
    month = int(value[4:6])
    day = int(value[6:8])
    hour = int(value[8:10])
    minute = int(value[10:12])
    if year < 1000 or month < 1 or month > 12 or day < 1 or day > 31 or hour < 0 or hour > 24 or minute < 0 or minute > 59:
        raise HTTPException(status_code=400, detail="Invalid time format: {}".format(value))


def date_range(start, end):
    date(start)
    date(end)
    if datetime.strptime(start, '%Y%m%d') > datetime.strptime(end, '%Y%m%d'):
        raise HTTPException(status_code=400, detail="Start date must be before end date.")


def time_range(start, end):
    time(start)
    time(end)
    if datetime.strptime(start, '%Y%m%d%H%M') >= datetime.strptime(end, '%Y%m%d%H%M'):
        raise HTTPException(status_code=400, detail="Start time must be before end time.")


def sunday(value):
    date(value)
    if datetime.strptime(value, '%Y%m%d').weekday() != 6:
        raise HTTPException(status_code=400, detail="Input date must be a Sunday. {} is not a Sunday.".format(value))


def latitude_list(value):
    elements = value.split(',')
    if len(elements) < 2:
        raise HTTPException(status_code=400, detail="Latitude input must be a comma seperated list of at least two latitudes.")
    for element in elements:
        try:
            number = float(element)
            if not (-90 <= number <= 90):
                raise HTTPException(status_code=400, detail="{} is not a valid latitude.".format(element))
        except Exception as e:
            raise HTTPException(status_code=400, detail="{} is not a valid latitude.".format(element))


def longitude_list(value):
    elements = value.split(',')
    if len(elements) < 2:
        raise HTTPException(status_code=400, detail="Longitude input must be a comma seperated list of at least two longitudes.")
    for element in elements:
        try:
            number = float(element)
            if not (-180 <= number <= 180):
                raise HTTPException(status_code=400, detail="{} is not a valid longitude.".format(element))
        except Exception as e:
            raise HTTPException(status_code=400, detail="{} is not a valid longitude.".format(element))
