from fastapi import FastAPI, Query, BackgroundTasks, HTTPException
from fastapi.responses import RedirectResponse, PlainTextResponse
from fastapi.middleware.gzip import GZipMiddleware

import sentry_sdk

from app import simulations
from app import meteoswiss
from app import bafu

import os

sentry_sdk.init(
    dsn="https://9b346d9bd9aa4309a18f5a47746b0a54@o1106970.ingest.sentry.io/4504402334777344",
    traces_sample_rate=1.0,
)

app = FastAPI(
    title="Alplakes API",
    description="API for the Alplakes project.",
    version="0.0.1",
    contact={
        "name": "James Runnalls",
        "email": "james.runnalls@eawag.ch",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

filesystem = "../filesystem"


@app.get("/")
def welcome():
    return {"Welcome to the Alplakes API from Eawag. Navigate to /docs or /redoc for documentation. For "
            "queries please contact James Runnalls (james.runnall@eawag.ch)."}


@app.get("/meteoswiss/cosmo/metadata", tags=["Meteoswiss"])
async def meteoswiss_cosmo_metadata():
    """
    JSON of all the available MeteoSwiss COSMO data.
    """
    return meteoswiss.get_cosmo_metadata(filesystem)


@app.get("/meteoswiss/cosmo/area/reanalysis/{model}/{start_date}/{end_date}/{ll_lat}/{ll_lng}/{ur_lat}/{ur_lng}",
         tags=["Meteoswiss"])
async def meteoswiss_cosmo_area_reanalysis(model: meteoswiss.CosmoReanalysis, start_date: str, end_date: str,
                                           ll_lat: float,
                                           ll_lng: float, ur_lat: float, ur_lng: float,
                                           variables: list[str] = Query(
                                               default=["T_2M", "U", "V", "GLOB", "RELHUM_2M", "PMSL", "CLCT"])):
    """
    Weather data from MeteoSwiss COSMO reanalysis for a rectangular area:
    - **cosmo**: select a COSMO product
        - VNXQ34 (reanalysis): Cosmo-1e 1 day deterministic
        - VNJK21 (reanalysis): Cosmo-1e 1 day ensemble forecast
    - **start_date**: start date "YYYYMMDD"
    - **end_date**: end date "YYYYMMDD"
    - **ll_lat**: Latitude of lower left corner of bounding box (WGS 84)
    - **ll_lng**: Longitude of lower left corner of bounding box (WGS 84)
    - **ur_lat**: Latitude of upper right corner of bounding box (WGS 84)
    - **ur_lng**: Longitude of upper right corner of bounding box (WGS 84)
    """
    meteoswiss.verify_cosmo_area_reanalysis(model, variables, start_date, end_date, ll_lat, ll_lng, ur_lat, ur_lng)
    return meteoswiss.get_cosmo_area_reanalysis(filesystem, model, variables, start_date, end_date, ll_lat, ll_lng, ur_lat,
                                           ur_lng)


@app.get("/meteoswiss/cosmo/area/forecast/{model}/{forecast_date}/{ll_lat}/{ll_lng}/{ur_lat}/{ur_lng}",
         tags=["Meteoswiss"])
async def meteoswiss_cosmo_area_forecast(model: meteoswiss.CosmoForecast, forecast_date: str,
                                         ll_lat: float, ll_lng: float, ur_lat: float, ur_lng: float,
                                         variables: list[str] = Query(
                                             default=["T_2M_MEAN", "U_MEAN", "V_MEAN", "GLOB_MEAN", "RELHUM_2M_MEAN",
                                                       "PMSL_MEAN", "CLCT_MEAN"])):
    """
    Weather data from MeteoSwiss COSMO forecasts for a rectangular area:
    - **cosmo**: select a COSMO product
        - VNXQ94 (forecast): Cosmo-1e 33 hour ensemble forecast
        - VNXZ32 (forecast): Cosmo-2e 5 day ensemble forecast
    - **forecast_date**: date of forecast "YYYYMMDD"
    - **ll_lat**: latitude of lower left corner of bounding box (WGS 84)
    - **ll_lng**: longitude of lower left corner of bounding box (WGS 84)
    - **ur_lat**: latitude of upper right corner of bounding box (WGS 84)
    - **ur_lng**: longitude of upper right corner of bounding box (WGS 84)
    """
    meteoswiss.verify_cosmo_area_forecast(model, variables, forecast_date, ll_lat, ll_lng, ur_lat, ur_lng)
    return meteoswiss.get_cosmo_area_forecast(filesystem, model, variables, forecast_date, ll_lat, ll_lng, ur_lat, ur_lng)


@app.get("/meteoswiss/cosmo/point/reanalysis/{model}/{start_date}/{end_date}/{lat}/{lng}", tags=["Meteoswiss"])
async def meteoswiss_cosmo_point_reanalysis(model: meteoswiss.CosmoReanalysis, start_date: str, end_date: str,
                                            lat: float, lng: float,
                                            variables: list[str] = Query(
                                                default=["T_2M", "U", "V", "GLOB", "RELHUM_2M", "PMSL", "CLCT"])):
    """
    Weather data from MeteoSwiss COSMO reanalysis for a single point:
    - **cosmo**: select a COSMO product
        - VNXQ34 (reanalysis): Cosmo-1e 1 day deterministic
        - VNJK21 (reanalysis): Cosmo-1e 1 day ensemble forecast
    - **start_date**: start date "YYYYMMDD"
    - **end_date**: end date "YYYYMMDD"
    - **lat**: Latitude of point (WGS 84)
    - **lng**: Longitude of point (WGS 84)
    """
    meteoswiss.verify_cosmo_point_reanalysis(model, variables, start_date, end_date, lat, lng)
    return meteoswiss.get_cosmo_point_reanalysis(filesystem, model, variables, start_date, end_date, lat, lng)


@app.get("/meteoswiss/cosmo/point/forecast/{model}/{forecast_date}/{lat}/{lng}", tags=["Meteoswiss"])
async def meteoswiss_cosmo_point_forecast(model: meteoswiss.CosmoForecast, forecast_date: str,
                                          lat: float, lng: float,
                                          variables: list[str] = Query(
                                              default=["T_2M_MEAN", "U_MEAN", "V_MEAN", "GLOB_MEAN", "RELHUM_2M_MEAN",
                                                       "PMSL_MEAN", "CLCT_MEAN"])):
    """
    Weather data from MeteoSwiss COSMO forecasts for a single point:
    - **cosmo**: select a COSMO product
        - VNXQ94 (forecast): Cosmo-1e 33 hour ensemble forecast
        - VNXZ32 (forecast): Cosmo-2e 5 day ensemble forecast
    - **forecast_date**: date of forecast "YYYYMMDD"
    - **lat**: latitude of point (WGS 84)
    - **lng**: longitude of point (WGS 84)
    """
    meteoswiss.verify_cosmo_point_forecast(model, variables, forecast_date, lat, lng)
    return meteoswiss.get_cosmo_point_forecast(filesystem, model, variables, forecast_date, lat, lng)


@app.get("/bafu/hydrodata/metadata", tags=["Bafu"])
async def bafu_hydrodata_metadata():
    """
    GEOJSON of all the available BAFU hydrodata.
    """
    return RedirectResponse("https://alplakes-eawag.s3.eu-central-1.amazonaws.com/static/bafu/bafu_hydrodata.json")


@app.get("/bafu/hydrodata/measured/{station_id}/{parameter}/{start_date}/{end_date}", tags=["Bafu"])
async def bafu_hydrodata_measured(station_id: int, parameter: str, start_date: str, end_date: str):
    """
    Measured hydrodata from Bafu:
    - **station_id**: 4 digit station identification code
    - **parameter**: parameter to retrieve (get list from /bafu/hydrodata/metadata)
    - **start_date**: start date "YYYYMMDD"
    - **end_date**: end date "YYYYMMDD"
    """
    bafu.verify_hydrodata_measured(station_id, parameter, start_date, end_date)
    return bafu.get_hydrodata_measured(filesystem, station_id, parameter, start_date, end_date)


@app.get("/bafu/hydrodata/predicted/{status}/{station_id}/{model}", tags=["Bafu"])
async def bafu_hydrodata_predicted(status: bafu.HydrodataPredicted, station_id: int, model: str):
    """
    Predicted hydrodata from Bafu:
    - **status**:
        - official: pqprevi-official
        - unofficial: pqprevi-unofficial
    - **station_id**: 4 digit station identification code
    - **model**: model to retrieve (get list from /bafu/hydrodata/metadata)
    """
    bafu.verify_hydrodata_predicted(status, station_id, model)
    return bafu.get_hydrodata_predicted(filesystem, status, station_id, model)


@app.get("/bafu/hydrodata/total_lake_inflow/metadata", tags=["Bafu"])
async def bafu_hydrodata_total_lake_inflow_metadata():
    """
    Metadata for the Bafu total lake inflow predictions.
    """
    return bafu.metadata_hydrodata_total_lake_inflow(filesystem)


@app.get("/bafu/hydrodata/total_lake_inflow/{lake}/{parameter}/{start_date}/{end_date}", tags=["Bafu"])
async def bafu_hydrodata_total_lake_inflow(lake, parameter: str, start_date: str, end_date: str):
    """
    Predicted total lake inflow from Bafu:
    - **lake**: lake name
    - **parameter**: parameter to retrieve (get list from /bafu/hydrodata/total_lake_inflow)
    """
    bafu.verify_hydrodata_total_lake_inflow(lake, parameter, start_date, end_date)
    return bafu.get_hydrodata_total_lake_inflow(filesystem, lake, parameter, start_date, end_date)


@app.get("/simulations/metadata", tags=["Simulations"])
async def simulations_metadata():
    """
    JSON of all the available Simulation data.
    """
    return simulations.get_metadata(filesystem)


@app.get("/simulations/layer/{model}/{lake}/{time}/{depth}", tags=["Simulations"])
async def simulations_layer(model: simulations.Models, lake: simulations.Lakes, time: str, depth: float):
    """
    Simulations results for a given lake simulation at a specific depth and time:
    - **model**: model name
    - **lake**: lake name
    - **time**: YYYYmmddHHMM (UTC) e.g. 9am 6th December 2022 > 202212060900
    - **depth**: depth of layer in meters
    """
    simulations.verify_simulations_layer(model, lake, time, depth)
    return simulations.get_simulations_layer(filesystem, model, lake, time, depth)


@app.get("/simulations/layer_alplakes/{model}/{lake}/{parameter}/{start}/{end}/{depth}", tags=["Simulations"], response_class=PlainTextResponse)
async def simulations_layer_alplakes(model: simulations.Models, lake: simulations.Lakes, parameter: simulations.Parameters, start: str, end: str, depth: float):
    """
    Parameters for a given lake simulation at a depth for period of time, formatted for the Alplakes website:
    - **model**: model name
    - **lake**: lake name
    - **parameter**: parameter name
    - **start**: YYYYmmddHHMM (UTC) e.g. 9am 6th December 2022 > 202212060900
    - **end**: YYYYmmddHHMM (UTC) e.g. 9am 6th December 2022 > 202212060900
    - **depth**: depth of layer in meters
    """
    simulations.verify_simulations_layer_alplakes(model, lake, parameter, start, end, depth)
    return simulations.get_simulations_layer_alplakes(filesystem, model, lake, parameter, start, end, depth)


@app.post("/simulations/notify", tags=["Simulations"])
async def simulations_notify(notification: simulations.Notification, background_tasks: BackgroundTasks):
    """
    Endpoint for notifying Alplakes API about new simulation results
    """
    n = notification.dict()
    if n["type"] == "new":
        accepted_buckets = ["https://alplakes-eawag.s3"]
        if n["value"][:25] in accepted_buckets:
            if n["model"] == "delft3d-flow":
                background_tasks.add_task(simulations.notify_new_delft3dflow, filesystem, n["model"], n["value"])
            else:
                raise HTTPException(status_code=400,
                                    detail="Notification system not configured for model: {}".format(n["model"]))
        else:
            raise HTTPException(status_code=500,
                                detail="Only new simulation in the Alplakes S3 Bucket are accepted.")
    else:
        raise HTTPException(status_code=400,
                            detail="The following notification type is not recognised: {}".format(n["type"]))
    return {"message": "Notification sent in the background"}
