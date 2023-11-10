from fastapi import FastAPI, Query, BackgroundTasks, HTTPException
from fastapi.responses import RedirectResponse, PlainTextResponse
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.cors import CORSMiddleware

import sentry_sdk

from app import simulations
from app import meteoswiss
from app import bafu

import os

sentry_sdk.init(
    dsn="https://9b346d9bd9aa4309a18f5a47746b0a54@o1106970.ingest.sentry.io/4504402334777344",
    traces_sample_rate=1.0,
)
origins = [
    "http://localhost:3000",
    "https://www.alplakes.eawag.ch",
    "https://master.d1x767yafo35xy.amplifyapp.com"
]

tags_metadata = [
    {
        "name": "meteoswiss",
        "description": "Geographical coverage [45.04116, 4.155702] to [47.98, 11.314313]",
    }
]

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

filesystem = "../filesystem"

internal = True

if os.getenv('EXTERNAL') is not None:
    internal = False


@app.get("/")
def welcome():
    return {"Welcome to the Alplakes API from Eawag. Navigate to /docs or /redoc for documentation. For "
            "queries please contact James Runnalls (james.runnall@eawag.ch)."}


if internal:
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
        return meteoswiss.get_cosmo_area_reanalysis(filesystem, model, variables, start_date, end_date, ll_lat, ll_lng,
                                                    ur_lat,
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
        return meteoswiss.get_cosmo_area_forecast(filesystem, model, variables, forecast_date, ll_lat, ll_lng, ur_lat,
                                                  ur_lng)

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

    @app.get("/meteoswiss/meteodata/metadata", tags=["Meteoswiss"])
    async def meteoswiss_meteodata_metadata():
        """
        GEOJSON of all Meteoswiss stations. Accessed from https://www.geocat.ch/geonetwork/srv/ger/catalog.search#/metadata/d9ea83de-b8d7-44e1-a7e3-3065699eb571
        """
        return RedirectResponse("https://alplakes-eawag.s3.eu-central-1.amazonaws.com/static/meteoswiss/meteoswiss_meteodata.json")

    @app.get("/meteoswiss/meteodata/measured/{station_id}/{parameter}/{start_date}/{end_date}", tags=["Meteoswiss"])
    async def meteoswiss_meteodata_measured(station_id: str, parameter: meteoswiss.MeteodataParameters, start_date: str, end_date: str):
        """
        Measured meteodata from Meteoswiss:
        - **station_id**: 3 digit station identification code
        - **parameter**: parameter to retrieve
            - pva200h0 (hPa): Vapour pressure 2 m above ground; hourly mean
            - gre000h0 (W/m²): Global radiation; hourly mean
            - tre200h0 (°C): Air temperature 2 m above ground; hourly mean
            - rre150h0 (mm): Precipitation; hourly total
            - fkl010h0 (m/s): Wind speed scalar; hourly mean
            - dkl010h0 (°): Wind direction; hourly mean
            - nto000d0 (%): Cloud cover; daily mean
        - **start_date**: start date "YYYYMMDD"
        - **end_date**: end date "YYYYMMDD"
        """
        meteoswiss.verify_meteodata_measured(station_id, parameter, start_date, end_date)
        return meteoswiss.get_meteodata_measured(filesystem, station_id, parameter, start_date, end_date)

if internal:
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


@app.get("/simulations/metadata/{model}/{lake}", tags=["Simulations"])
async def simulations_metadata_lake(model: simulations.Models, lake: simulations.Lakes):
    """
    JSON of the available Simulation data.
    """
    simulations.verify_metadata_lake(model, lake)
    return simulations.get_metadata_lake(filesystem, model, lake)


@app.get("/simulations/file/{model}/{lake}/{sunday}", tags=["Simulations"])
async def simulations_file(model: simulations.Models, lake: simulations.Lakes, sunday: str):
    """
    Simulation file for a given lake simulation for the selected week:
    - **model**: model name
    - **lake**: lake name
    - **sunday**: YYYYmmdd (UTC) always a sunday.
    """
    simulations.verify_simulations_file(model, lake, sunday)
    return simulations.get_simulations_file(filesystem, model, lake, sunday)


@app.get("/simulations/point/{model}/{lake}/{start}/{end}/{depth}/{latitude}/{longitude}", tags=["Simulations"])
async def simulations_point(model: simulations.Models, lake: simulations.Lakes, start: str, end: str, depth: float,
                            latitude: float, longitude: float):
    """
    Data for a given lake simulation at a specific time and location:
    - **model**: model name
    - **lake**: lake name
    - **start**: YYYYmmddHHMM (UTC) e.g. 9am 6th December 2022 > 202212060900
    - **end**: YYYYmmddHHMM (UTC) e.g. 9am 6th December 2022 > 202212060900
    - **depth**: depth of layer in meters
    - **latitude**: Latitude (WGS84)
    - **longitude**: Longitude (WGS84)
    """
    simulations.verify_simulations_point(model, lake, start, end, depth, latitude, longitude)
    return simulations.get_simulations_point(filesystem, model, lake, start, end, depth, latitude, longitude)


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


@app.get("/simulations/layer_alplakes/{model}/{lake}/{parameter}/{start}/{end}/{depth}", tags=["Simulations"],
         response_class=PlainTextResponse)
async def simulations_layer_alplakes(model: simulations.Models, lake: simulations.Lakes,
                                     parameter: simulations.Parameters, start: str, end: str, depth: float):
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


@app.get("/simulations/layer/average_temperature/{model}/{lake}/{start}/{end}/{depth}", tags=["Simulations"])
async def simulations_layer_average_temperature(model: simulations.Models, lake: simulations.Lakes,
                                                start: str, end: str, depth: float):
    """
    Parameters for a given lake simulation at a depth for period of time, formatted for the Alplakes website:
    - **model**: model name
    - **lake**: lake name
    - **start**: YYYYmmddHHMM (UTC) e.g. 9am 6th December 2022 > 202212060900
    - **end**: YYYYmmddHHMM (UTC) e.g. 9am 6th December 2022 > 202212060900
    - **depth**: depth of layer in meters
    """
    return simulations.get_simulations_layer_average_temperature(filesystem, model, lake, start, end, depth)


@app.get("/simulations/profile/{model}/{lake}/{datetime}/{latitude}/{longitude}", tags=["Simulations"])
async def simulations_profile(model: simulations.Models, lake: simulations.Lakes, datetime: str, latitude: float,
                              longitude: float):
    """
    Profile for a given lake simulation at a specific time and location:
    - **model**: model name
    - **lake**: lake name
    - **datetime**: YYYYmmddHHMM (UTC) e.g. 9am 6th December 2022 > 202212060900
    - **latitude**: Latitude (WGS84)
    - **longitude**: Longitude (WGS84)
    """
    simulations.verify_simulations_profile(model, lake, datetime, latitude, longitude)
    return simulations.get_simulations_profile(filesystem, model, lake, datetime, latitude, longitude)


@app.get("/simulations/transect/{model}/{lake}/{datetime}/{latitude_list}/{longitude_list}", tags=["Simulations"])
async def simulations_transect(model: simulations.Models, lake: simulations.Lakes, datetime: str, latitude_list: str,
                               longitude_list: str):
    """
    Transect for a given lake simulation at a specific time:
    - **model**: model name
    - **lake**: lake name
    - **datetime**: YYYYmmddHHMM (UTC) e.g. 9am 6th December 2022 > 202212060900
    - **latitude_list**: Comma separated list of latitude (WGS84) values (minimum 2) e.g. /46.37,46.54/
    - **longitude_list**: Comma separated list of longitude (WGS84) values (minimum 2) e.g. /6.56,6.54/
    """
    simulations.verify_simulations_transect(model, lake, datetime, latitude_list, longitude_list)
    return simulations.get_simulations_transect(filesystem, model, lake, datetime, latitude_list, longitude_list)


@app.get("/simulations/transect/{model}/{lake}/{start}/{end}/{latitude_list}/{longitude_list}", tags=["Simulations"])
async def simulations_transect_period(model: simulations.Models, lake: simulations.Lakes, start: str, end: str,
                                      latitude_list: str, longitude_list: str):
    """
    Transect for a given lake simulation over a given period:
    - **model**: model name
    - **lake**: lake name
    - **start**: YYYYmmddHHMM (UTC) e.g. 9am 6th December 2022 > 202212060900
    - **end**: YYYYmmddHHMM (UTC) e.g. 9am 6th December 2022 > 202212060900
    - **latitude_list**: Comma separated list of latitude (WGS84) values (minimum 2) e.g. /46.37,46.54/
    - **longitude_list**: Comma separated list of longitude (WGS84) values (minimum 2) e.g. /6.56,6.54/
    """
    simulations.verify_simulations_transect_period(model, lake, start, end, latitude_list, longitude_list)
    return simulations.get_simulations_transect_period(filesystem, model, lake, start, end, latitude_list, longitude_list)


@app.get("/simulations/depthtime/{model}/{lake}/{start}/{end}/{latitude}/{longitude}", tags=["Simulations"])
async def simulations_depth_time(model: simulations.Models, lake: simulations.Lakes, start: str, end: str,
                                 latitude: float, longitude: float):
    """
    Depth time data for a specific location over a given time period:
    - **model**: model name
    - **lake**: lake name
    - **start**: YYYYmmddHHMM (UTC) e.g. 9am 6th December 2022 > 202212060900
    - **end**: YYYYmmddHHMM (UTC) e.g. 9am 6th December 2022 > 202212060900
    - **latitude**: Latitude (WGS84)
    - **longitude**: Longitude (WGS84)
    """
    simulations.verify_simulations_depthtime(model, lake, start, end, latitude, longitude)
    return simulations.get_simulations_depthtime(filesystem, model, lake, start, end, latitude, longitude)
