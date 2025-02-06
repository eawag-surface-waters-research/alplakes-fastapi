from fastapi import FastAPI, Query, Request, Depends, Path, BackgroundTasks, Response, HTTPException
from fastapi.responses import RedirectResponse, PlainTextResponse, JSONResponse, FileResponse
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Union, Any
import sentry_sdk

from app import simulations, meteoswiss, bafu, remotesensing, insitu, validate, thredds, geosphere, arso, mistral

import os

sentry_sdk.init(
    dsn="https://9b346d9bd9aa4309a18f5a47746b0a54@o1106970.ingest.sentry.io/4504402334777344",
    traces_sample_rate=1.0,
)
origins = [
    "http://localhost:3000",
    "https://www.alplakes.eawag.ch",
    "https://www.datalakes-eawag.ch",
    "https://www.datalakes.eawag.ch",
    "https://master.d1x767yafo35xy.amplifyapp.com",
    "https://pr-55.d21l70hd8m002c.amplifyapp.com"
]

description = """
Alplakes API connects you to lake products produced by the [SURF](https://www.eawag.ch/en/department/surf/) department at [EAWAG](https://www.eawag.ch).

This includes terabytes of simulation data and remote sensing products. The API supports geospatial and temporal queries, allowing access to subsets of the data for easier handling.

This API serves as the backend for the website [www.alplakes.eawag.ch](http://www.alplakes.eawag.ch).

### Disclaimer

The **Alplakes API** is provided "as is," without any guarantees regarding the accuracy, completeness, or timeliness of the data. While we strive to ensure data quality, users are responsible for verifying information before making any decisions based on it.

Additionally, we cannot guarantee continuous availability of the API. Service disruptions or maintenance periods may occur, and users should expect intermittent downtime.

### Get in Touch

For bug reports, collaboration requests, or to join our mailing list for updates, feel free to [get in touch](mailto:james.runnalls@eawag.ch).
"""

app = FastAPI(
    title="Alplakes API",
    description=description,
    version="2.0.0",
    contact={
        "name": "James Runnalls",
        "email": "james.runnalls@eawag.ch",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    }
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

filesystem = "filesystem"

internal = True

if os.getenv('EXTERNAL') is not None:
    internal = False


@app.exception_handler(ValueError)
async def value_error_exception_handler(request: Request, exc: ValueError):
    print(exc)
    raise
    return JSONResponse(
        status_code=500,
        content={"message": "Server processing error - please check your inputs. The developer has been notified, for "
                            "updates on bug fixes please contact James Runnalls (james.runnall@eawag.ch)"})


@app.get("/", include_in_schema=False)
def welcome():
    return {"Welcome to the Alplakes API from Eawag. Navigate to /docs or /redoc for documentation. For "
            "queries please contact James Runnalls (james.runnall@eawag.ch)."}


if internal:
    @app.get("/meteoswiss/cosmo/metadata", tags=["Meteoswiss"], response_model=List[meteoswiss.Metadata])
    async def meteoswiss_cosmo_metadata():
        """
        Metadata for available MeteoSwiss COSMO data.
        """
        return meteoswiss.get_cosmo_metadata(filesystem)

    @app.get("/meteoswiss/cosmo/area/reanalysis/{model}/{start_date}/{end_date}/{ll_lat}/{ll_lng}/{ur_lat}/{ur_lng}", tags=["Meteoswiss"], response_model=meteoswiss.ResponseModel2D)
    async def meteoswiss_cosmo_area_reanalysis(model: meteoswiss.CosmoReanalysis,
                                               start_date: str = validate.path_date(description="The start date in YYYYmmdd format"),
                                               end_date: str = validate.path_date(description="The end date in YYYYmmdd format"),
                                               ll_lat: float = validate.path_latitude(example="46.49", description="Latitude of lower left corner of bounding box (WGS 84)"),
                                               ll_lng: float = validate.path_longitude(example="6.65", description="Longitude of lower left corner of bounding box (WGS 84)"),
                                               ur_lat: float = validate.path_latitude(example="46.51", description="Latitude of upper right corner of bounding box (WGS 84)"),
                                               ur_lng: float = validate.path_longitude(example="6.67", description="Longitude of upper right corner of bounding box (WGS 84)"),
                                               variables: list[str] = Query(default=["T_2M", "U", "V", "GLOB", "RELHUM_2M", "PMSL", "CLCT"])):
        """
        Weather data from MeteoSwiss COSMO reanalysis for a rectangular bounding box.

        Available models:
        - VNXQ34 (reanalysis): Cosmo-1e 1 day deterministic

        ⚠️ **Warning**: If your request returns a 502 timeout error reduce the period you are requesting.
        For longer durations, it is recommended to make multiple requests with shorter intervals between them.
        """
        validate.date_range(start_date, end_date)
        return meteoswiss.get_cosmo_area_reanalysis(filesystem, model, variables, start_date, end_date, ll_lat, ll_lng, ur_lat, ur_lng)

    @app.get("/meteoswiss/cosmo/area/forecast/{model}/{forecast_date}/{ll_lat}/{ll_lng}/{ur_lat}/{ur_lng}", tags=["Meteoswiss"], response_model=meteoswiss.ResponseModel2D)
    async def meteoswiss_cosmo_area_forecast(model: meteoswiss.CosmoForecast,
                                             forecast_date: str = validate.path_date(description="The forecast date in YYYYmmdd format"),
                                             ll_lat: float = validate.path_latitude(example="46.49", description="Latitude of lower left corner of bounding box (WGS 84)"),
                                             ll_lng: float = validate.path_longitude(example="6.65", description="Longitude of lower left corner of bounding box (WGS 84)"),
                                             ur_lat: float = validate.path_latitude(example="46.51", description="Latitude of upper right corner of bounding box (WGS 84)"),
                                             ur_lng: float = validate.path_longitude(example="6.67", description="Longitude of upper right corner of bounding box (WGS 84)"),
                                             variables: list[str] = Query(default=["T_2M", "U", "V", "GLOB", "RELHUM_2M", "PMSL", "CLCT"])):
        """
        Weather data from MeteoSwiss COSMO forecasts for a rectangular bounding box.

        Available models:
        - VNXZ32 (forecast): Cosmo-2e 5 day ensemble forecast
        - VNXQ94 (forecast): Cosmo-1e 33 hour ensemble forecast
        """
        validate.date(forecast_date)
        return meteoswiss.get_cosmo_area_forecast(filesystem, model, variables, forecast_date, ll_lat, ll_lng, ur_lat, ur_lng)

    @app.get("/meteoswiss/cosmo/point/reanalysis/{model}/{start_date}/{end_date}/{lat}/{lng}", tags=["Meteoswiss"], response_model=meteoswiss.ResponseModel1D)
    async def meteoswiss_cosmo_point_reanalysis(model: meteoswiss.CosmoReanalysis,
                                                start_date: str = validate.path_date(description="The start date in YYYYmmdd format"),
                                                end_date: str = validate.path_date(description="The end date in YYYYmmdd format"),
                                                lat: float = validate.path_latitude(),
                                                lng: float = validate.path_longitude(),
                                                variables: list[str] = Query(default=["T_2M", "U", "V", "GLOB", "RELHUM_2M", "PMSL", "CLCT"])):
        """
        Weather data from MeteoSwiss COSMO reanalysis for a single point.

        Available models:
        - VNXQ34 (reanalysis): Cosmo-1e 1 day deterministic

        ⚠️ **Warning**: If your request returns a 502 timeout error reduce the period you are requesting.
        For longer durations, it is recommended to make multiple requests with shorter intervals between them.
        """
        validate.date_range(start_date, end_date)
        return meteoswiss.get_cosmo_point_reanalysis(filesystem, model, variables, start_date, end_date, lat, lng)

    @app.get("/meteoswiss/cosmo/point/forecast/{model}/{forecast_date}/{lat}/{lng}", tags=["Meteoswiss"], response_model=meteoswiss.ResponseModel1D)
    async def meteoswiss_cosmo_point_forecast(model: meteoswiss.CosmoForecast,
                                              forecast_date: str = validate.path_date(description="The forecast date in YYYYmmdd format"),
                                              lat: float = validate.path_latitude(),
                                              lng: float = validate.path_longitude(),
                                              variables: list[str] = Query(
                                                  default=["T_2M", "U", "V", "GLOB", "RELHUM_2M",
                                                           "PMSL", "CLCT"])):
        """
        Weather data from MeteoSwiss COSMO forecasts for a single point.

        Available models:
        - VNXQ94 (forecast): Cosmo-1e 33 hour ensemble forecast
        - VNXZ32 (forecast): Cosmo-2e 5 day ensemble forecast
        """
        validate.date(forecast_date)
        return meteoswiss.get_cosmo_point_forecast(filesystem, model, variables, forecast_date, lat, lng)


    @app.get("/meteoswiss/icon/metadata", tags=["Meteoswiss"], response_model=List[meteoswiss.Metadata])
    async def meteoswiss_icon_metadata():
        """
        Metadata for available MeteoSwiss ICON data.
        """
        return meteoswiss.get_icon_metadata(filesystem)


    @app.get("/meteoswiss/icon/area/reanalysis/{model}/{start_date}/{end_date}/{ll_lat}/{ll_lng}/{ur_lat}/{ur_lng}", tags=["Meteoswiss"], response_model=meteoswiss.ResponseModel2D)
    async def meteoswiss_icon_area_reanalysis(model: meteoswiss.IconReanalysis,
                                               start_date: str = validate.path_date(
                                                   description="The start date in YYYYmmdd format", example="20240729"),
                                               end_date: str = validate.path_date(
                                                   description="The end date in YYYYmmdd format", example="20240729"),
                                               ll_lat: float = validate.path_latitude(example="46.49",
                                                                                      description="Latitude of lower left corner of bounding box (WGS 84)"),
                                               ll_lng: float = validate.path_longitude(example="6.65",
                                                                                       description="Longitude of lower left corner of bounding box (WGS 84)"),
                                               ur_lat: float = validate.path_latitude(example="46.51",
                                                                                      description="Latitude of upper right corner of bounding box (WGS 84)"),
                                               ur_lng: float = validate.path_longitude(example="6.67",
                                                                                       description="Longitude of upper right corner of bounding box (WGS 84)"),
                                               variables: list[str] = Query(
                                                   default=["T_2M", "U", "V", "GLOB", "RELHUM_2M", "PMSL", "CLCT"])):
        """
        Weather data from MeteoSwiss ICON reanalysis for a rectangular bounding box.

        Available models:
        - kenda-ch1 (reanalysis): 1 day deterministic

        ⚠️ **Warning**: If your request returns a 502 timeout error reduce the period you are requesting.
        For longer durations, it is recommended to make multiple requests with shorter intervals between them.
        """
        validate.date_range(start_date, end_date)
        return meteoswiss.get_icon_area_reanalysis(filesystem, model, variables, start_date, end_date, ll_lat, ll_lng,
                                                    ur_lat, ur_lng)

    @app.get("/meteoswiss/icon/area/forecast/{model}/{forecast_date}/{ll_lat}/{ll_lng}/{ur_lat}/{ur_lng}", tags=["Meteoswiss"], response_model=meteoswiss.ResponseModel2D)
    async def meteoswiss_icon_area_forecast(model: meteoswiss.IconForecast,
                                             forecast_date: str = validate.path_date(
                                                 description="The forecast date in YYYYmmdd format", example="20240703"),
                                             ll_lat: float = validate.path_latitude(example="46.49",
                                                                                    description="Latitude of lower left corner of bounding box (WGS 84)"),
                                             ll_lng: float = validate.path_longitude(example="6.65",
                                                                                     description="Longitude of lower left corner of bounding box (WGS 84)"),
                                             ur_lat: float = validate.path_latitude(example="46.51",
                                                                                    description="Latitude of upper right corner of bounding box (WGS 84)"),
                                             ur_lng: float = validate.path_longitude(example="6.67",
                                                                                     description="Longitude of upper right corner of bounding box (WGS 84)"),
                                             variables: list[str] = Query(
                                                 default=["T_2M", "U", "V", "GLOB",
                                                          "RELHUM_2M", "PMSL", "CLCT"])):
        """
        Weather data from MeteoSwiss ICON forecasts for a rectangular bounding box.

        Available models:
        - icon-ch2-eps (forecast):  ICON-CH2-EPS 5 day ensemble forecast
        - icon-ch1-eps (forecast):  ICON-CH1-EPS 33 hour ensemble forecast
        """
        validate.date(forecast_date)
        return meteoswiss.get_icon_area_forecast(filesystem, model, variables, forecast_date, ll_lat, ll_lng, ur_lat,
                                                  ur_lng)

    @app.get("/meteoswiss/icon/point/reanalysis/{model}/{start_date}/{end_date}/{lat}/{lng}", tags=["Meteoswiss"], response_model=meteoswiss.ResponseModel1D)
    async def meteoswiss_icon_point_reanalysis(model: meteoswiss.IconReanalysis,
                                               start_date: str = validate.path_date(
                                                   description="The start date in YYYYmmdd format", example="20240729"),
                                               end_date: str = validate.path_date(
                                                   description="The end date in YYYYmmdd format", example="20240729"),
                                               lat: float = validate.path_latitude(),
                                               lng: float = validate.path_longitude(),
                                               variables: list[str] = Query(
                                                   default=["T_2M", "U", "V", "GLOB", "RELHUM_2M", "PMSL", "CLCT"])):
        """
        Weather data from MeteoSwiss ICON reanalysis for a single point.

        Available models:
        - kenda-ch1 (reanalysis): 1 day deterministic

        ⚠️ **Warning**: If your request returns a 502 timeout error reduce the period you are requesting.
        For longer durations, it is recommended to make multiple requests with shorter intervals between them.
        """
        validate.date_range(start_date, end_date)
        return meteoswiss.get_icon_point_reanalysis(filesystem, model, variables, start_date, end_date, lat, lng)

    @app.get("/meteoswiss/icon/point/forecast/{model}/{forecast_date}/{lat}/{lng}", tags=["Meteoswiss"], response_model=meteoswiss.ResponseModel1D)
    async def meteoswiss_icon_point_forecast(model: meteoswiss.IconForecast,
                                              forecast_date: str = validate.path_date(
                                                  description="The forecast date in YYYYmmdd format", example="20240703"),
                                              lat: float = validate.path_latitude(),
                                              lng: float = validate.path_longitude(),
                                              variables: list[str] = Query(
                                                  default=["T_2M", "U", "V", "GLOB",
                                                           "RELHUM_2M",
                                                           "PMSL", "CLCT"])):
        """
        Weather data from MeteoSwiss ICON forecasts for a single point.

        Available models:
        - icon-ch2-eps (forecast):  ICON-CH2-EPS 5 day ensemble forecast
        - icon-ch1-eps (forecast):  ICON-CH1-EPS 33 hour ensemble forecast
        """
        validate.date(forecast_date)
        return meteoswiss.get_icon_point_forecast(filesystem, model, variables, forecast_date, lat, lng)

    @app.get("/meteoswiss/meteodata/metadata", tags=["Meteoswiss"], response_class=RedirectResponse, response_description="Redirect to a GeoJSON file")
    async def meteoswiss_meteodata_metadata():
        """
        Metadata for all Meteoswiss metreological stations.

        Accessed from
        https://www.geocat.ch/geonetwork/srv/ger/catalog.search#/metadata/d9ea83de-b8d7-44e1-a7e3-3065699eb571.
        Last updated 27.10.2023 05:30.
        """
        return RedirectResponse("https://alplakes-eawag.s3.eu-central-1.amazonaws.com/static/meteoswiss/meteoswiss_meteodata.json")


    @app.get("/meteoswiss/meteodata/metadata/{station_id}", tags=["Meteoswiss"], response_model=meteoswiss.ResponseModelMeteoMeta)
    async def meteoswiss_meteodata_station_metadata(
            station_id: str = Path(..., regex=r"^[a-zA-Z0-9]{3,6}$", title="Station ID", example="PUY",
                                   description="3 digit station identification code")):
        """
        Meteorological data from the automatic measuring network of MeteoSwiss.

        Metadata for a specific station.
        """
        return meteoswiss.get_meteodata_station_metadata(filesystem, station_id)

    @app.get("/meteoswiss/meteodata/measured/{station_id}/{start_date}/{end_date}", tags=["Meteoswiss"], response_model=meteoswiss.ResponseModelMeteo)
    async def meteoswiss_meteodata_measured(station_id: str = Path(..., regex=r"^[a-zA-Z0-9]{3,6}$", title="Station ID", example="PUY", description="3 digit station identification code"),
                                            start_date: str = validate.path_date(description="The start date in YYYYmmdd format"),
                                            end_date: str = validate.path_date(description="The end date in YYYYmmdd format"),
                                            variables: list[str] = Query(
                                                  default=["vapour_pressure", "global_radiation", "air_temperature", "precipitation", "wind_speed",
                                                           "wind_direction"])):
        """
        Meteorological data from the automatic measuring network of MeteoSwiss.

        See metadata endpoint for available variables.

        ⚠️ **Warning**: If your request returns a 502 timeout error reduce the period you are requesting.
        For longer durations, it is recommended to make multiple requests with shorter intervals between them.
        """
        validate.date_range(start_date, end_date)
        return meteoswiss.get_meteodata_measured(filesystem, station_id, variables, start_date, end_date)

if internal:
    @app.get("/thredds/meteodata/metadata", tags=["Thredds"], response_class=RedirectResponse,
             response_description="Redirect to a GeoJSON file")
    async def thredds_meteodata_metadata():
        """
        Metadata for all available Thredds metreological stations (France).
        """
        return RedirectResponse(
            "https://alplakes-eawag.s3.eu-central-1.amazonaws.com/static/thredds/thredds_meteodata.json")


    @app.get("/thredds/meteodata/metadata/{station_id}", tags=["Thredds"],
             response_model=thredds.ResponseModelMeteoMeta)
    async def thredds_meteodata_station_metadata(
            station_id: str = Path(..., title="Station ID", example="73329001",
                                   description="Station identification code")):
        """
        Meteorological data from the automatic measuring network of Thredds (France).

        Metadata for a specific station.
        """
        return thredds.get_meteodata_station_metadata(filesystem, station_id)


    @app.get("/thredds/meteodata/measured/{station_id}/{start_date}/{end_date}", tags=["Thredds"],
             response_model=thredds.ResponseModelMeteo)
    async def thredds_meteodata_measured(
            station_id: str = Path(..., title="Station ID", example="73329001",
                                   description="Station identification code"),
            start_date: str = validate.path_date(description="The start date in YYYYmmdd format"),
            end_date: str = validate.path_date(description="The end date in YYYYmmdd format"),
            variables: list[str] = Query(
                default=["air_temperature", "relative_humidity", "wind_speed", "wind_direction", "precipitation",
                         "global_radiation"])):
        """
        Meteorological data from the automatic measuring network of Thredds (France).

        See metadata endpoint for available variables.

        ⚠️ **Warning**: If your request returns a 502 timeout error reduce the period you are requesting.
        For longer durations, it is recommended to make multiple requests with shorter intervals between them.
        """
        validate.date_range(start_date, end_date)
        return thredds.get_meteodata_measured(filesystem, station_id, variables, start_date, end_date)

if internal:
    @app.get("/arso/meteodata/metadata", tags=["ARSO"], response_class=RedirectResponse,
             response_description="Redirect to a GeoJSON file")
    async def arso_meteodata_metadata():
        """
        Metadata for all available ARSO metreological stations (Slovenia).
        """
        return RedirectResponse(
            "https://alplakes-eawag.s3.eu-central-1.amazonaws.com/static/arso/arso_meteodata.json")


    @app.get("/arso/meteodata/metadata/{station_id}", tags=["ARSO"],
             response_model=arso.ResponseModelMeteoMeta)
    async def arso_meteodata_station_metadata(
            station_id: str = Path(..., title="Station ID", example="2213",
                                   description="Station identification code")):
        """
        Meteorological data from the automatic measuring network of ARSO (Slovenia).

        Metadata for a specific station.
        """
        return arso.get_meteodata_station_metadata(filesystem, station_id)


    @app.get("/arso/meteodata/measured/{station_id}/{start_date}/{end_date}", tags=["ARSO"],
             response_model=arso.ResponseModelMeteo)
    async def arso_meteodata_measured(
            station_id: str = Path(..., title="Station ID", example="2213",
                                   description="Station identification code"),
            start_date: str = validate.path_date(description="The start date in YYYYmmdd format"),
            end_date: str = validate.path_date(description="The end date in YYYYmmdd format"),
            variables: list[str] = Query(
                default=["air_pressure", "air_temperature", "relative_humidity", "wind_speed", "wind_direction", "precipitation",
                         "global_radiation"])):
        """
        Meteorological data from the automatic measuring network of ARSO (Slovenia).

        See metadata endpoint for available variables.

        ⚠️ **Warning**: If your request returns a 502 timeout error reduce the period you are requesting.
        For longer durations, it is recommended to make multiple requests with shorter intervals between them.
        """
        validate.date_range(start_date, end_date)
        return arso.get_meteodata_measured(filesystem, station_id, variables, start_date, end_date)

if internal:
    @app.get("/mistral/meteodata/metadata", tags=["Mistral"], response_class=RedirectResponse,
             response_description="Redirect to a GeoJSON file")
    async def mistral_meteodata_metadata():
        """
        Metadata for all available Mistral metreological stations (Italy).
        """
        return RedirectResponse(
            "https://alplakes-eawag.s3.eu-central-1.amazonaws.com/static/mistral/mistral_meteodata.json")


    @app.get("/mistral/meteodata/metadata/{station_id}", tags=["Mistral"],
             response_model=mistral.ResponseModelMeteoMeta)
    async def mistral_meteodata_station_metadata(
            station_id: str = Path(..., title="Station ID", example="trn196",
                                   description="Station identification code")):
        """
        Meteorological data from the automatic measuring network of Mistral (Italy).

        Metadata for a specific station.
        """
        return mistral.get_meteodata_station_metadata(filesystem, station_id)


    @app.get("/mistral/meteodata/measured/{station_id}/{start_date}/{end_date}", tags=["Mistral"],
             response_model=mistral.ResponseModelMeteo)
    async def mistral_meteodata_measured(
            station_id: str = Path(..., title="Station ID", example="trn196",
                                   description="Station identification code"),
            start_date: str = validate.path_date(description="The start date in YYYYmmdd format"),
            end_date: str = validate.path_date(description="The end date in YYYYmmdd format"),
            variables: list[str] = Query(
                default=["air_temperature", "relative_humidity", "wind_speed", "wind_direction", "global_radiation"])):
        """
        Meteorological data from the automatic measuring network of Mistral (Italy).

        See metadata endpoint for available variables.

        ⚠️ **Warning**: If your request returns a 502 timeout error reduce the period you are requesting.
        For longer durations, it is recommended to make multiple requests with shorter intervals between them.
        """
        validate.date_range(start_date, end_date)
        return mistral.get_meteodata_measured(filesystem, station_id, variables, start_date, end_date)


if internal:
    @app.get("/geosphere/meteodata/metadata", tags=["GeoSphere"], response_class=RedirectResponse,
             response_description="Redirect to a GeoJSON file")
    async def geosphere_meteodata_metadata():
        """
        Metadata for all available GeoSphere metreological stations (Austria).
        """
        return RedirectResponse(
            "https://alplakes-eawag.s3.eu-central-1.amazonaws.com/static/geosphere/geosphere_meteodata.json")


    @app.get("/geosphere/meteodata/metadata/{station_id}", tags=["GeoSphere"],
             response_model=geosphere.ResponseModelMeteoMeta)
    async def geosphere_meteodata_station_metadata(
            station_id: str = Path(..., title="Station ID", example="6512",
                                   description="Station identification code")):
        """
        Meteorological data from the automatic measuring network of GeoSphere (Austria).

        Metadata for a specific station.
        """
        return geosphere.get_meteodata_station_metadata(filesystem, station_id)


    @app.get("/geosphere/meteodata/measured/{station_id}/{start_date}/{end_date}", tags=["GeoSphere"],
             response_model=geosphere.ResponseModelMeteo)
    async def geosphere_meteodata_measured(
            station_id: str = Path(..., title="Station ID", example="6512",
                                   description="Station identification code"),
            start_date: str = validate.path_date(description="The start date in YYYYmmdd format"),
            end_date: str = validate.path_date(description="The end date in YYYYmmdd format"),
            variables: list[str] = Query(
                default=["air_temperature", "relative_humidity", "wind_speed", "wind_direction", "precipitation",
                         "global_radiation", "air_pressure"]),
            resample: Union[geosphere.ResampleOptions, None] = None):
        """
        Meteorological data from the automatic measuring network of GeoSphere (Austria).

        See metadata endpoint for available variables.

        ⚠️ **Warning**: If your request returns a 502 timeout error reduce the period you are requesting.
        For longer durations, it is recommended to make multiple requests with shorter intervals between them.
        """
        validate.date_range(start_date, end_date)
        return geosphere.get_meteodata_measured(filesystem, station_id, variables, start_date, end_date, resample)

if internal:
    @app.get("/bafu/hydrodata/metadata", tags=["Bafu"], response_class=RedirectResponse, response_description="Redirect to a GeoJSON file")
    async def bafu_hydrodata_metadata():
        """
        Metadata for all the available BAFU hydrodata.

        Accessed from https://www.geocat.ch/geonetwork/srv/ger/catalog.search#/metadata/bd377507-c61d-4fe2-bf81-9605b405d8ad
        """
        return RedirectResponse("https://alplakes-eawag.s3.eu-central-1.amazonaws.com/static/bafu/bafu_hydrodata.json")


    @app.get("/bafu/hydrodata/metadata/{station_id}", tags=["Bafu"], response_model=bafu.ResponseModelMeta)
    async def bafu_hydrodata_station_metadata(station_id: str = Path(..., regex=r"^\d{4}$", title="Station ID", example=2009, description="4 digit station identification code")):
        """
        Hydrological data from the automatic measuring network of Bafu.

        Metadata for a specific station.
        """
        return bafu.get_hydrodata_station_metadata(filesystem, station_id)

    @app.get("/bafu/hydrodata/measured/{station_id}/{variable}/{start_date}/{end_date}", tags=["Bafu"], response_model=bafu.ResponseModel)
    async def bafu_hydrodata_measured(station_id: str = Path(..., regex=r"^\d{4}$", title="Station ID", example=2009, description="4 digit station identification code"),
                                      variable: str = Path(..., title="Variable", example="AbflussPneumatikunten", description="Variable"),
                                      start_date: str = validate.path_date(description="The start date in YYYYmmdd format"),
                                      end_date: str = validate.path_date(description="The end date in YYYYmmdd format"),
                                      resample: Union[bafu.ResampleOptions, None] = None):
        """
        Hydrological data from the automatic measuring network of Bafu.

        All station id's and the available variables
        can be access from the metadata endpoint. Data is a mix of hourly data (pre-2022) and 5-min data and can be
        resampled to hourly or daily.

        ⚠️ **Warning**: If your request returns a 502 timeout error reduce the period you are requesting.
        For longer durations, it is recommended to make multiple requests with shorter intervals between them.
        """
        validate.date_range(start_date, end_date)
        return bafu.get_hydrodata_measured(filesystem, station_id, variable, start_date, end_date, resample)

if internal:
    @app.get("/insitu/secchi/metadata", tags=["Insitu"], response_model=List[insitu.Metadata])
    async def insitu_secchi_metadata():
        """
        Insitu Secchi depth measurements from assorted monitoring programs

        Metadata for all included lakes.
        """
        return insitu.get_insitu_secchi_metadata(filesystem)


    @app.get("/insitu/secchi/{lake}", tags=["Insitu"], response_model=insitu.ResponseModel)
    async def insitu_secchi_lake(lake: str = Path(..., title="Lake", example="geneva", description="Lake name")):
        """
        Insitu Secchi depth measurements from assorted monitoring programs

        Full timeseries for requested lake.
        """
        return insitu.get_insitu_secchi_lake(filesystem, lake)


@app.get("/simulations/metadata", tags=["3D Simulations"], response_model=List[simulations.Metadata])
async def simulations_metadata():
    """
    Metadata for all available 3D simulations.
    """
    return simulations.get_metadata(filesystem)


@app.get("/simulations/metadata/{model}/{lake}", tags=["3D Simulations"], response_model=simulations.MetadataLake)
async def simulations_metadata_lake(model: simulations.Models = Path(..., title="Model", description="Model name"),
                                    lake: simulations.Lakes = Path(..., title="Lake", description="Lake name")):
    """
    Metadata for a specific 3D model and lake.
    """
    return simulations.get_metadata_lake(filesystem, model, lake)


@app.get("/simulations/file/{model}/{lake}/{sunday}", tags=["3D Simulations"], response_class=FileResponse)
async def simulations_file(model: simulations.Models = Path(..., title="Model", description="Model name"),
                           lake: simulations.Lakes = Path(..., title="Lake", description="Lake name"),
                           sunday: str = validate.path_date(description="The Sunday in YYYYmmdd format", example="20230402")):
    """
    NetCDF simulation results for a one-week period.

    Simulation results are in their native format from the simulations see below for guides describing the output files.

    | Model        | Link                                                                                                                                                                                                                               |
    |--------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
    | delft3d-flow | [https://github.com/eawag-surface-waters-research/alplakes-simulations/blob/master/static/delft3d-flow/OUTPUT.md](https://github.com/eawag-surface-waters-research/alplakes-simulations/blob/master/static/delft3d-flow/OUTPUT.md) |

    ⚠️ **Warning:** Trying to run this endpoint in the docs is likely to crash your browser due to the size of the files.
    Call the url directly for downloading.

    For example: [https://alplakes-api.eawag.ch/simulations/file/delft3d-flow/geneva/20230101](https://alplakes-api.eawag.ch/simulations/file/delft3d-flow/geneva/20230101)
    """
    validate.sunday(sunday)
    path = os.path.join(filesystem, "media/simulations", model, "results", lake, "{}.nc".format(sunday))
    if not os.path.isfile(path):
        raise HTTPException(status_code=400, detail="Apologies data is not available for {} on the week beginning {}".format(model, sunday))
    return FileResponse(path, media_type="application/nc", filename="{}_{}_{}.nc".format(model, lake, sunday))


@app.get("/simulations/point/{model}/{lake}/{start_time}/{end_time}/{depth}/{lat}/{lng}", tags=["3D Simulations"], response_model=simulations.ResponseModelPoint)
async def simulations_point(model: simulations.Models = Path(..., title="Model", description="Model name"),
                            lake: simulations.Lakes = Path(..., title="Lake", description="Lake name"),
                            start_time: str = validate.path_time(description="The start time in YYYYmmddHHMM format (UTC)", example="202304050300"),
                            end_time: str = validate.path_time(description="The end time in YYYYmmddHHMM format (UTC)", example="202304112300"),
                            depth: float = validate.path_depth(),
                            lat: float = validate.path_latitude(),
                            lng: float = validate.path_longitude(),
                            variables: list[str] = Query(default=["temperature", "velocity"])):
    """
    Simulation timeseries for a given location and depth over a defined time period.

    ⚠️ **Warning**: If your request returns a 502 timeout error reduce the period you are requesting.
    For longer durations, it is recommended to make multiple requests with shorter intervals between them.
    """
    validate.time_range(start_time, end_time)
    return simulations.get_simulations_point(filesystem, model, lake, start_time, end_time, depth, lat, lng, variables)


@app.get("/simulations/layer/{model}/{lake}/{time}/{depth}", tags=["3D Simulations"], response_model=simulations.ResponseModelLayer)
async def simulations_layer(model: simulations.Models = Path(..., title="Model", description="Model name"),
                            lake: simulations.Lakes = Path(..., title="Lake", description="Lake name"),
                            time: str = validate.path_time(description="The start time in YYYYmmddHHMM format (UTC)", example="202304050300"),
                            depth: float = validate.path_depth(),
                            variables: list[str] = Query(default=["temperature", "velocity"])):
    """
    Simulation results for an entire depth layer at a specific time.
    """
    validate.time(time)
    return simulations.get_simulations_layer(filesystem, model, lake, time, depth, variables)


@app.get("/simulations/layer_alplakes/{model}/{lake}/{variable}/{start_time}/{end_time}/{depth}", tags=["3D Simulations"],
         response_class=PlainTextResponse, include_in_schema=internal)
async def simulations_layer_alplakes(model: simulations.Models = Path(..., title="Model", description="Model name"),
                                     lake: simulations.Lakes = Path(..., title="Lake", description="Lake name"),
                                     variable: simulations.Variables = Path(..., title="Variable", description="Variable"),
                                     start_time: str = validate.path_time(description="The start time in YYYYmmddHHMM format (UTC)", example="202304050300"),
                                     end_time: str = validate.path_time(description="The end time in YYYYmmddHHMM format (UTC)", example="202304112300"),
                                     depth: float = validate.path_depth()):
    """
    Plain text formatted variables for a depth layer over a specific time range. Reduces file size compared to layer outputs.

    ⚠️ **Warning:** This endpoint is designed for supplying data to the Alplakes website. The output is **not** self-explanatory.
    """
    validate.time_range(start_time, end_time)
    return simulations.get_simulations_layer_alplakes(filesystem, model, lake, variable, start_time, end_time, depth)


@app.get("/simulations/layer/average_temperature/{model}/{lake}/{start_time}/{end_time}/{depth}", tags=["3D Simulations"], response_model=simulations.ResponseModelAverageLayer)
async def simulations_layer_average_temperature(model: simulations.Models = Path(..., title="Model", description="Model name"),
                                                lake: simulations.Lakes = Path(..., title="Lake", description="Lake name"),
                                                start_time: str = validate.path_time(description="The start time in YYYYmmddHHMM format (UTC)", example="202304050300"),
                                                end_time: str = validate.path_time(description="The end time in YYYYmmddHHMM format (UTC)", example="202304112300"),
                                                depth: float = validate.path_depth()):
    """
    Mean temperature at a specified depth for a given time period.

    ⚠️ **Warning**: If your request returns a 502 timeout error reduce the period you are requesting.
    For longer durations, it is recommended to make multiple requests with shorter intervals between them.
    """
    validate.time_range(start_time, end_time)
    return simulations.get_simulations_layer_average_temperature(filesystem, model, lake, start_time, end_time, depth)


@app.get("/simulations/layer/average_bottom_temperature/{model}/{lake}/{start_time}/{end_time}", tags=["3D Simulations"], response_model=simulations.ResponseModelAverageBottom)
async def simulations_layer_average_temperature(model: simulations.Models = Path(..., title="Model", description="Model name"),
                                                lake: simulations.Lakes = Path(..., title="Lake", description="Lake name"),
                                                start_time: str = validate.path_time(description="The start time in YYYYmmddHHMM format (UTC)", example="202304050300"),
                                                end_time: str = validate.path_time(description="The end time in YYYYmmddHHMM format (UTC)", example="202304112300")):
    """
    Mean bottom temperature for a given time period.

    ⚠️ **Warning**: If your request returns a 502 timeout error reduce the period you are requesting.
    For longer durations, it is recommended to make multiple requests with shorter intervals between them.
    """
    validate.time_range(start_time, end_time)
    return simulations.get_simulations_average_bottom_temperature(filesystem, model, lake, start_time, end_time)


@app.get("/simulations/profile/{model}/{lake}/{time}/{lat}/{lng}", tags=["3D Simulations"], response_model=simulations.ResponseModelProfile)
async def simulations_profile(model: simulations.Models = Path(..., title="Model", description="Model name"),
                              lake: simulations.Lakes = Path(..., title="Lake", description="Lake name"),
                              time: str = validate.path_time(),
                              lat: float = validate.path_latitude(),
                              lng: float = validate.path_longitude(),
                              variables: list[str] = Query(default=["temperature", "velocity"])):
    """
    Vertical profile for a specific **time** and location.
    """
    validate.time(time)
    return simulations.get_simulations_profile(filesystem, model, lake, time, lat, lng, variables)


@app.get("/simulations/depthtime/{model}/{lake}/{start_time}/{end_time}/{lat}/{lng}", tags=["3D Simulations"], response_model=simulations.ResponseModelDepthTime)
async def simulations_depth_time(model: simulations.Models = Path(..., title="Model", description="Model name"),
                                 lake: simulations.Lakes = Path(..., title="Lake", description="Lake name"),
                                 start_time: str = validate.path_time(description="The start time in YYYYmmddHHMM format (UTC)", example="202304050300"),
                                 end_time: str = validate.path_time(description="The end time in YYYYmmddHHMM format (UTC)", example="202304112300"),
                                 lat: float = validate.path_latitude(),
                                 lng: float = validate.path_longitude(),
                                 variables: list[str] = Query(default=["temperature", "velocity"])):
    """
    Vertical profile for a specific **period** and location.

    ⚠️ **Warning**: If your request returns a 502 timeout error reduce the period you are requesting.
    For longer durations, it is recommended to make multiple requests with shorter intervals between them.
    """
    validate.time_range(start_time, end_time)
    return simulations.get_simulations_depthtime(filesystem, model, lake, start_time, end_time, lat, lng, variables)


@app.get("/simulations/transect/{model}/{lake}/{time}/{lats}/{lngs}", tags=["3D Simulations"], response_model=simulations.ResponseModelTransect)
async def simulations_transect(model: simulations.Models = Path(..., title="Model", description="Model name"),
                               lake: simulations.Lakes = Path(..., title="Lake", description="Lake name"),
                               time: str = validate.path_time(),
                               lats: str = Path(..., title="Lats", description="Comma separated list of latitudes (WGS84), minimum 2", example="46.37,46.54"),
                               lngs: str = Path(..., title="Lngs", description="Comma separated list of longitudes (WGS84), minimum 2", example="6.56,6.54"),
                               variables: list[str] = Query(default=["temperature", "velocity"])):
    """
    Transect for a specific time.
    """
    validate.latitude_list(lats)
    validate.longitude_list(lngs)
    return simulations.get_simulations_transect(filesystem, model, lake, time, lats, lngs, variables)


@app.get("/simulations/transect/{model}/{lake}/{start_time}/{end_time}/{lats}/{lngs}", tags=["3D Simulations"], response_model=simulations.ResponseModelTransectPeriod)
async def simulations_transect_period(model: simulations.Models = Path(..., title="Model", description="Model name"),
                                      lake: simulations.Lakes = Path(..., title="Lake", description="Lake name"),
                                      start_time: str = validate.path_time( description="The start time in YYYYmmddHHMM format (UTC)", example="202304050300"),
                                      end_time: str = validate.path_time( description="The end time in YYYYmmddHHMM format (UTC)", example="202304051200"),
                                      lats: str = Path(..., title="Lats", description="Comma separated list of latitudes (WGS84), minimum 2", example="46.37,46.54"),
                                      lngs: str = Path(..., title="Lngs", description="Comma separated list of longitudes (WGS84), minimum 2", example="6.56,6.54"),
                                      variables: list[str] = Query(default=["temperature", "velocity"])):
    """
    Transect for a specific period.

    ⚠️ **Warning**: If your request returns a 502 timeout error reduce the period you are requesting.
    For longer durations, it is recommended to make multiple requests with shorter intervals between them.
    """
    validate.latitude_list(lats)
    validate.longitude_list(lngs)
    validate.time_range(start_time, end_time)
    return simulations.get_simulations_transect_period(filesystem, model, lake, start_time, end_time, lats, lngs, variables)


@app.get("/simulations/1d/metadata", tags=["1D Simulations"], response_model=List[simulations.Metadata1D])
async def one_dimensional_simulations_metadata():
    """
    Metadata for all available 1D simulations.

    Doesn't include available variables, using the endpoint below for full variables list.
    """
    return simulations.get_one_dimensional_metadata(filesystem)


@app.get("/simulations/1d/metadata/{model}/{lake}", tags=["1D Simulations"], response_model=simulations.MetadataLake1DDetail)
async def one_dimensional_simulations_metadata_lake(
        model: simulations.OneDimensionalModels = Path(..., title="Model", description="Model name"),
        lake: str = Path(..., title="Lake", description="Lake key", example="aegeri")):
    """
    Available 1D simulation data for a specific lake and model. Includes information on available variables.
    """
    return simulations.get_one_dimensional_metadata_lake(filesystem, model, lake)


@app.get("/simulations/1d/file/{model}/{lake}/{month}", tags=["1D Simulations"], response_class=FileResponse)
async def one_dimensional_simulations_file(
        model: simulations.OneDimensionalModels = Path(..., title="Model", description="Model name"),
        lake: str = Path(..., title="Lake", description="Lake key", example="aegeri"),
        month: str = validate.path_month(description="The month in YYYYmm format")):
    """
    Full model output in NetCDF for a given month.

    Metadata in the file describes the available variables.
    """
    path = os.path.join(filesystem, "media/1dsimulations", model, "results", lake, "{}.nc".format(month))
    if not os.path.isfile(path):
        raise HTTPException(status_code=400, detail="Apologies data is not available for {} on the month beginning {}".format(model,month))
    return FileResponse(path, media_type="application/nc", filename="{}_{}_{}.nc".format(model, lake, month))

@app.get("/simulations/1d/point/{model}/{lake}/{start_time}/{end_time}/{depth}", tags=["1D Simulations"], response_model=simulations.ResponseModel1DPoint)
async def one_dimensional_simulations_point(
        model: simulations.OneDimensionalModels = Path(..., title="Model", description="Model name"),
        lake: str = Path(..., title="Lake", description="Lake key", example="aegeri"),
        start_time: str = validate.path_time(description="The start time in YYYYmmddHHMM format (UTC)", example="202405050300"),
        end_time: str = validate.path_time(description="The end time in YYYYmmddHHMM format (UTC)", example="202406072300"),
        depth: float = validate.path_depth(),
        resample: Union[simulations.SimstratResampleOptions, None] = None,
        variables: list[str] = Query(default=["T"])):
    """
    Timeseries for a given location and depth.

    See the metadata endpoints for list of available variables.

     ⚠️ **Warning**: If your request returns a 502 timeout error reduce the period you are requesting.
        For longer durations, it is recommended to make multiple requests with shorter intervals between them.
    """
    validate.time_range(start_time, end_time)
    return simulations.get_one_dimensional_point(filesystem, model, lake, start_time, end_time, depth, variables, resample)


@app.get("/simulations/1d/profile/{model}/{lake}/{time}", tags=["1D Simulations"], response_model=simulations.ResponseModel1DProfile)
async def one_dimensional_simulations_profile(
        model: simulations.OneDimensionalModels = Path(..., title="Model", description="Model name"),
        lake: str = Path(..., title="Lake", description="Lake key", example="aegeri"),
        time: str = validate.path_time(description="The time in YYYYmmddHHMM format (UTC)", example="202405050300"),
        variables: list[str] = Query(default=["T"])):
    """
    Vertical profile for a specific **time**.
    """
    validate.time(time)
    return simulations.get_one_dimensional_profile(filesystem, model, lake, time, variables)


@app.get("/simulations/1d/depthtime/{model}/{lake}/{start_time}/{end_time}", tags=["1D Simulations"], response_model=simulations.ResponseModel1DDepthTime)
async def one_dimensional_simulations_depth_time(
        model: simulations.OneDimensionalModels = Path(..., title="Model", description="Model name"),
        lake: str = Path(..., title="Lake", description="Lake key", example="aegeri"),
        start_time: str = validate.path_time(description="The start time in YYYYmmddHHMM format (UTC)", example="202405050300"),
        end_time: str = validate.path_time(description="The end time in YYYYmmddHHMM format (UTC)", example="202406072300"),
        variables: list[str] = Query(default=["T"])):
    """
    Vertical profile for a specific **period**.

     ⚠️ **Warning**: If your request returns a 502 timeout error reduce the period you are requesting.
        For longer durations, it is recommended to make multiple requests with shorter intervals between them.
    """
    validate.time_range(start_time, end_time)
    return simulations.get_one_dimensional_depth_time(filesystem, model, lake, start_time, end_time, variables)


@app.get("/simulations/1d/doy/metadata", tags=["1D Simulations"], response_model=List[simulations.Metadata1DDOY])
async def one_dimensional_simulations_day_of_year_metadata():
    """
    Metadata for all available DOY products
    """
    return simulations.get_one_dimensional_day_of_year_metadata(filesystem)


@app.get("/simulations/1d/doy/{model}/{lake}/{variable}/{depth}", tags=["1D Simulations"], response_model=simulations.ResponseModel1DDOY)
async def one_dimensional_simulations_day_of_year(
        model: simulations.OneDimensionalModels = Path(..., title="Model", description="Model name"),
        lake: str = Path(..., title="Lake", description="Lake key", example="aegeri"),
        variable: str = Path(..., title="Variable", description="Variable", example="T"),
        depth: float = validate.path_depth(example="1.0")):
    """
    Day of year statistics for a given variable.

    ⚠️ **Warning:** Only available once computed using the write endpoint, see metadata for available products
    """
    return simulations.get_one_dimensional_day_of_year(filesystem, model, lake, variable, depth)


if internal:
    @app.get("/simulations/1d/doy/write/{model}/{lake}/{variable}/{depth}", tags=["1D Simulations"], responses={202: {"description": "Computing DOY"}})
    async def write_one_dimensional_simulations_day_of_year(
            background_tasks: BackgroundTasks,
            model: simulations.OneDimensionalModels = Path(..., title="Model", description="Model name"),
            lake: str = Path(..., title="Lake", description="Lake key", example="aegeri"),
            variable: str = Path(..., title="Variable", description="Variable", example="T"),
            depth: float = validate.path_depth()):
        """
        Compute day of year statistics for a given variable.

        ⚠️ **Warning:** Processing is slow due to the large volumes of data, once it has completed data is available from the doy
        endpoint.
        """
        background_tasks.add_task(simulations.write_one_dimensional_day_of_year, filesystem, model, lake, variable, depth)
        return Response(content="Computing DOY", status_code=202)


@app.get("/remotesensing/metadata", tags=["Remote Sensing"], response_class=RedirectResponse, response_description="Redirect to a GeoJSON file")
async def remote_sensing_metadata():
    """
    Directory of remote sensing product types, organised by lake, satellite and variable.
    """
    return RedirectResponse("https://eawagrs.s3.eu-central-1.amazonaws.com/alplakes/metadata/summary.json")


@app.get("/remotesensing/products/{lake}/{satellite}/{variable}", tags=["Remote Sensing"], response_class=RedirectResponse, response_description="Redirect to a GeoJSON file")
async def remote_sensing_products(lake: str = Path(..., title="Lake", description="Lake name", example="geneva"),
                                  satellite: remotesensing.Satellites = Path(..., title="Satellite", description="Satellite name"),
                                  variable: str = Path(..., title="Variable", description="Variable", example="chla")):
    """
    Metadata for full time series of remote sensing products for a given lake, satellite and variable.
    See /remotesensing/metadata for input options.
    """
    return RedirectResponse("https://eawagrs.s3.eu-central-1.amazonaws.com/alplakes/metadata/{}/{}/{}_public.json".format(satellite, lake, variable))
