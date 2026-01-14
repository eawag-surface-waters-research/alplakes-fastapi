import os
import requests
from enum import Enum
from fastapi import HTTPException
from pydantic import RootModel, field_validator
from typing import List
from datetime import datetime, timezone

from app import functions

class Satellites(str, Enum):
    sentinel3 = "sentinel3"
    sentinel2 = "sentinel2"
    collection = "collection"


class ResponseModelSatelliteMetadata(functions.TimeBaseModel):
    time: datetime
    full_tile: str
    lake_subset: str
    valid_pixels: str
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

class ResponseModelSatellite(RootModel):
    root: List[ResponseModelSatelliteMetadata]

    def __iter__(self):
        return iter(self.root)

    def __getitem__(self, item):
        return self.root[item]


def get_remote_sensing_products(lake, satellite, variable, min_date, max_date, valid_pixels):
    url = "https://eawagrs.s3.eu-central-1.amazonaws.com/alplakes/metadata/{}/{}/{}_public.json".format(satellite, lake, variable)
    try:
        response = requests.get(url)
        data = response.json()
    except:
        raise HTTPException(status_code=400, detail="Unable to collect metadata for that combination of lake, satellite and variable.")

    return [
        format_response(d, satellite, lake) for d in data
        if (min_date is None or int(d['datetime'].replace("T", "")[:12]) >= int(min_date))
           and (max_date is None or int(d['datetime'].replace("T", "")[:12]) <= int(max_date))
           and (valid_pixels is None or int(d['valid_pixels'].replace("%", "")) >= int(valid_pixels))
    ]

def format_response(data, satellite, lake):
    product = data["url"].split("/")[-1].replace(".tif", "")
    lake_subset = "https://eawagrs.s3.eu-central-1.amazonaws.com/alplakes/cropped/{}/{}/{}_{}.tif".format(satellite, lake, product, lake)
    time = datetime.strptime(data["datetime"], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
    return { "time": time,
             "full_tile": data["url"],
             "lake_subset": lake_subset,
             "valid_pixels": data["valid_pixels"] }