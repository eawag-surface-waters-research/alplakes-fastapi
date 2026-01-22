import json
import requests
import numpy as np
from enum import Enum
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from pydantic import RootModel, field_validator
from typing import List
from datetime import datetime, timezone
import rasterio
from rasterio.session import AWSSession
from rasterio.windows import Window

from app import functions

aws_session = AWSSession(
    requester_pays=False,
    aws_unsigned=True
)

rasterio_env = rasterio.Env(
    session=aws_session,
    GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif,.tiff',
    GDAL_HTTP_TIMEOUT='30',
    GDAL_HTTP_MAX_RETRY='3',
    VSI_CACHE=True,
    VSI_CACHE_SIZE=10485760  # 10MB cache
)

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
        metadata = response.json()
    except:
        raise HTTPException(status_code=400, detail="Unable to collect metadata for that combination of lake, satellite and variable.")

    metadata = list({m['url']: m for m in metadata}.values())

    metadata =  [
        format_response(d, satellite, lake) for d in metadata
        if (min_date is None or int(d['datetime'].replace("T", "")[:12]) >= int(min_date))
           and (max_date is None or int(d['datetime'].replace("T", "")[:12]) <= int(max_date))
           and (valid_pixels is None or int(d['valid_pixels'].replace("%", "")) >= int(valid_pixels))
    ]

    return metadata


def format_response(data, satellite, lake):
    product = data["url"].split("/")[-1].replace(".tif", "")
    lake_subset = "https://eawagrs.s3.eu-central-1.amazonaws.com/alplakes/cropped/{}/{}/{}_{}.tif".format(satellite, lake, product, lake)
    time = datetime.strptime(data["datetime"], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
    return { "time": time,
             "full_tile": data["url"],
             "lake_subset": lake_subset,
             "valid_pixels": data["valid_pixels"] }


def get_remote_sensing_timeseries(lake, satellite, variable, lat, lng, window, min_date, max_date, valid_pixels, stream):
    url = "https://eawagrs.s3.eu-central-1.amazonaws.com/alplakes/metadata/{}/{}/{}_public.json".format(satellite, lake, variable)
    try:
        response = requests.get(url)
        metadata = response.json()
    except:
        raise HTTPException(status_code=400, detail="Unable to collect metadata for that combination of lake, satellite and variable.")

    metadata = list({m['url']: m for m in metadata}.values())

    metadata = [
        format_response(d, satellite, lake) for d in metadata
        if (min_date is None or int(d['datetime'].replace("T", "")[:12]) >= int(min_date))
           and (max_date is None or int(d['datetime'].replace("T", "")[:12]) <= int(max_date))
           and (valid_pixels is None or int(d['valid_pixels'].replace("%", "")) >= int(valid_pixels))
    ]

    metadata = sorted(metadata, key=lambda x: x['time'], reverse=True)

    if stream:
        return StreamingResponse(
            generate_timeseries_stream(lng, lat, window, metadata),
            media_type="application/x-ndjson"
        )
    else:
        return generate_timeseries(lng, lat, window, metadata)


def generate_timeseries_stream(lng, lat, window, metadata):
    for file_info in metadata:
        value = get_pixel_value(file_info['lake_subset'], lng, lat, window)
        if value is not None:
            data_point = {
                "time": file_info["time"].isoformat(),
                "value": value,
            }
            yield json.dumps(data_point) + "\n"


def generate_timeseries(lng, lat, window, metadata):
    data_points = []
    for file_info in metadata:
        value = get_pixel_value(file_info['lake_subset'], lng, lat, window)
        if value is not None:
            data_points.append({
                "time": file_info["time"],
                "value": value,
            })
    return data_points


def get_pixel_value(url, lng, lat, window_radius):
    with rasterio_env:
        try:
            with rasterio.open(url) as src:
                py, px = src.index(lng, lat)

                col_off = max(0, px - window_radius)
                row_off = max(0, py - window_radius)
                width = min(window_radius * 2 + 1, src.width - col_off)
                height = min(window_radius * 2 + 1, src.height - row_off)

                if px < 0 or px >= src.width or py < 0 or py >= src.height:
                    return None

                window = Window(col_off, row_off, width, height)
                data = src.read(1, window=window)

                mask = np.isnan(data)
                if src.nodata is not None:
                    mask = mask | (data == src.nodata)
                valid_data = data[~mask]
                if valid_data.size == 0:
                    return None

                return {
                    'mean': float(np.mean(valid_data)),
                    'median': float(np.median(valid_data)),
                    'min': float(np.min(valid_data)),
                    'max': float(np.max(valid_data)),
                    'std': float(np.std(valid_data)),
                    'count': int(valid_data.size),
                }

        except Exception:
            return None
