from fastapi import HTTPException
from fastapi.responses import StreamingResponse

import requests


def get_metadata():
    try:
        response = requests.get("https://eawagrs.s3.eu-central-1.amazonaws.com/metadata/metadata.json", stream=True)
        response.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=str(e))
    return StreamingResponse(response.iter_content(chunk_size=1024), media_type='application/json')


def get_lake_products(lake, satellite, parameter):
    try:
        response = requests.get("https://eawagrs.s3.eu-central-1.amazonaws.com/metadata/{}/{}_{}_public.json"
                                .format(satellite, lake, parameter), stream=True)
        response.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=str(e))
    return StreamingResponse(response.iter_content(chunk_size=1024), media_type='application/json')
