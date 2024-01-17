from fastapi import FastAPI, Query, Request, Depends, Path
from fastapi.responses import RedirectResponse, PlainTextResponse, JSONResponse
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.cors import CORSMiddleware

import sentry_sdk

from app import simulations, meteoswiss, bafu, remotesensing, validate

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
    "https://master.d1x767yafo35xy.amplifyapp.com"
]

app = FastAPI(
    title="Alplakes API",
    description="Alplakes API provides researchers and the public programmatic access to "
                "historical and forecasted simulation data from lakes across the alpine region. The API supports both "
                "geospatial and temporal queries. It is the backend for the website www.alplakes.eawag.ch. For bug "
                "reports, collaborations or requests please get in touch.",
    version="1.0.0",
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

filesystem = "../filesystem"

internal = True

if os.getenv('EXTERNAL') is not None:
    internal = False


@app.exception_handler(ValueError)
async def value_error_exception_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=500,
        content={"message": "Server processing error - please check your inputs. The developer has been notified, for "
                            "updates on bug fixes please contact James Runnalls (james.runnall@eawag.ch)"})


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

    @app.get("/meteoswiss/cosmo/area/reanalysis/{model}/{start_date}/{end_date}/{ll_lat}/{ll_lng}/{ur_lat}/{ur_lng}", tags=["Meteoswiss"])
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
        - VNJK21 (reanalysis): Cosmo-1e 1 day ensemble forecast
        """
        validate.date_range(start_date, end_date)
        return meteoswiss.get_cosmo_area_reanalysis(filesystem, model, variables, start_date, end_date, ll_lat, ll_lng, ur_lat, ur_lng)

    @app.get("/meteoswiss/cosmo/area/forecast/{model}/{forecast_date}/{ll_lat}/{ll_lng}/{ur_lat}/{ur_lng}", tags=["Meteoswiss"])
    async def meteoswiss_cosmo_area_forecast(model: meteoswiss.CosmoForecast,
                                             forecast_date: str = validate.path_date(description="The forecast date in YYYYmmdd format"),
                                             ll_lat: float = validate.path_latitude(example="46.49", description="Latitude of lower left corner of bounding box (WGS 84)"),
                                             ll_lng: float = validate.path_longitude(example="6.65", description="Longitude of lower left corner of bounding box (WGS 84)"),
                                             ur_lat: float = validate.path_latitude(example="46.51", description="Latitude of upper right corner of bounding box (WGS 84)"),
                                             ur_lng: float = validate.path_longitude(example="6.67", description="Longitude of upper right corner of bounding box (WGS 84)"),
                                             variables: list[str] = Query(default=["T_2M_MEAN", "U_MEAN", "V_MEAN", "GLOB_MEAN", "RELHUM_2M_MEAN", "PMSL_MEAN", "CLCT_MEAN"])):
        """
        Weather data from MeteoSwiss COSMO forecasts for a rectangular bounding box.

        Available models:
        - VNXZ32 (forecast): Cosmo-2e 5 day ensemble forecast
        - VNXQ94 (forecast): Cosmo-1e 33 hour ensemble forecast
        """
        validate.date(forecast_date)
        return meteoswiss.get_cosmo_area_forecast(filesystem, model, variables, forecast_date, ll_lat, ll_lng, ur_lat, ur_lng)

    @app.get("/meteoswiss/cosmo/point/reanalysis/{model}/{start_date}/{end_date}/{lat}/{lng}", tags=["Meteoswiss"])
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
        - VNJK21 (reanalysis): Cosmo-1e 1 day ensemble forecast

        """
        validate.date_range(start_date, end_date)
        return meteoswiss.get_cosmo_point_reanalysis(filesystem, model, variables, start_date, end_date, lat, lng)

    @app.get("/meteoswiss/cosmo/point/forecast/{model}/{forecast_date}/{lat}/{lng}", tags=["Meteoswiss"])
    async def meteoswiss_cosmo_point_forecast(model: meteoswiss.CosmoForecast,
                                              forecast_date: str = validate.path_date(description="The forecast date in YYYYmmdd format"),
                                              lat: float = validate.path_latitude(),
                                              lng: float = validate.path_longitude(),
                                              variables: list[str] = Query(
                                                  default=["T_2M_MEAN", "U_MEAN", "V_MEAN", "GLOB_MEAN", "RELHUM_2M_MEAN",
                                                           "PMSL_MEAN", "CLCT_MEAN"])):
        """
        Weather data from MeteoSwiss COSMO forecasts for a single point.

        Available models:
        - VNXQ94 (forecast): Cosmo-1e 33 hour ensemble forecast
        - VNXZ32 (forecast): Cosmo-2e 5 day ensemble forecast
        """
        validate.date(forecast_date)
        return meteoswiss.get_cosmo_point_forecast(filesystem, model, variables, forecast_date, lat, lng)

    @app.get("/meteoswiss/meteodata/metadata", tags=["Meteoswiss"])
    async def meteoswiss_meteodata_metadata():
        """
        GEOJSON of all Meteoswiss stations. Accessed from https://www.geocat.ch/geonetwork/srv/ger/catalog.search#/metadata/d9ea83de-b8d7-44e1-a7e3-3065699eb571
        """
        return RedirectResponse("https://alplakes-eawag.s3.eu-central-1.amazonaws.com/static/meteoswiss/meteoswiss_meteodata.json")

    @app.get("/meteoswiss/meteodata/measured/{station_id}/{parameter}/{start_date}/{end_date}", tags=["Meteoswiss"])
    async def meteoswiss_meteodata_measured(station_id: str = Path(..., regex=r"^[a-zA-Z]{3}$", title="Station ID", example="ABO", description="3 digit station identification code"),
                                            parameter: meteoswiss.MeteodataParameters = Path(..., title="Parameter", description="Meteoswiss parameter"),
                                            start_date: str = validate.path_date(description="The start date in YYYYmmdd format"),
                                            end_date: str = validate.path_date(description="The end date in YYYYmmdd format")):
        """
        Meteorological data from the automatic measuring network of MeteoSwiss.

        Available parameters:
        - pva200h0 (hPa): Vapour pressure 2 m above ground; hourly mean
        - gre000h0 (W/m²): Global radiation; hourly mean
        - tre200h0 (°C): Air temperature 2 m above ground; hourly mean
        - rre150h0 (mm): Precipitation; hourly total
        - fkl010h0 (m/s): Wind speed scalar; hourly mean
        - dkl010h0 (°): Wind direction; hourly mean
        - nto000d0 (%): Cloud cover; daily mean
        """
        validate.date_range(start_date, end_date)
        return meteoswiss.get_meteodata_measured(filesystem, station_id, parameter, start_date, end_date)

if internal:
    @app.get("/bafu/hydrodata/metadata", tags=["Bafu"])
    async def bafu_hydrodata_metadata():
        """
        GEOJSON of all the available BAFU hydrodata.
        """
        return RedirectResponse("https://alplakes-eawag.s3.eu-central-1.amazonaws.com/static/bafu/bafu_hydrodata.json")

    @app.get("/bafu/hydrodata/measured/{station_id}/{parameter}/{start_date}/{end_date}", tags=["Bafu"])
    async def bafu_hydrodata_measured(station_id: str = Path(..., regex=r"^\d{4}$", title="Station ID", example=2009, description="4 digit station identification code"),
                                      parameter: str = Path(..., title="Parameter", example="AbflussPneumatikunten", description="Parameter"),
                                      start_date: str = validate.path_date(description="The start date in YYYYmmdd format"),
                                      end_date: str = validate.path_date(description="The end date in YYYYmmdd format")):
        """
        Hydrological data from the automatic measuring network of Bafu.

        All station id's and the available parameters can be access from the metadata endpoint.
        """
        validate.date_range(start_date, end_date)
        return bafu.get_hydrodata_measured(filesystem, station_id, parameter, start_date, end_date)

    @app.get("/bafu/hydrodata/predicted/{status}/{station_id}/{model}", tags=["Bafu"])
    async def bafu_hydrodata_predicted(status: bafu.HydrodataPredicted = Path(..., title="Status", description="Publication status"),
                                       station_id: str = Path(..., regex=r"^\d{4}$", title="Station ID", example=2009, description="4 digit station identification code"),
                                       model: str = Path(..., title="Model", description="Predictive model", example="C1E_Med")):
        """
        Hydrological predictions from the Bafu predictive river models. Data is only available for the most recent prediction.

        All station id's and the available parameters can be access from the metadata endpoint.

        Publication status:
        - unofficial: pqprevi-unofficial
        - official: pqprevi-official
        """
        return bafu.get_hydrodata_predicted(filesystem, status, station_id, model)

    @app.get("/bafu/hydrodata/total_lake_inflow/metadata", tags=["Bafu"])
    async def bafu_hydrodata_total_lake_inflow_metadata():
        """
        Metadata for the Bafu total lake inflow predictions.
        """
        return bafu.metadata_hydrodata_total_lake_inflow(filesystem)

    @app.get("/bafu/hydrodata/total_lake_inflow/{lake}/{parameter}/{start_date}/{end_date}", tags=["Bafu"])
    async def bafu_hydrodata_total_lake_inflow(lake: str = Path(..., title="Lake", example="Lac_Leman", description="Lake name"),
                                               parameter: str = Path(..., title="Parameter", example="C1E_Med", description="Parameter"),
                                               start_date: str = validate.path_date(description="The start date in YYYYmmdd format"),
                                               end_date: str = validate.path_date(description="The end date in YYYYmmdd format")):
        """
        Predicted total lake inflow from Bafu.
        """
        validate.date_range(start_date, end_date)
        return bafu.get_hydrodata_total_lake_inflow(filesystem, lake, parameter, start_date, end_date)


@app.get("/simulations/metadata", tags=["Simulations"])
async def simulations_metadata():
    """
    JSON of all the available simulation data.
    """
    return simulations.get_metadata(filesystem)


@app.get("/simulations/metadata/{model}/{lake}", tags=["Simulations"])
async def simulations_metadata_lake(model: simulations.Models = Path(..., title="Model", description="Model name"),
                                    lake: simulations.Lakes = Path(..., title="Lake", description="Lake name")):
    """
    JSON of the available simulation data for a specific lake and model.
    """
    return simulations.get_metadata_lake(filesystem, model, lake)


@app.get("/simulations/file/{model}/{lake}/{sunday}", tags=["Simulations"])
async def simulations_file(model: simulations.Models = Path(..., title="Model", description="Model name"),
                           lake: simulations.Lakes = Path(..., title="Lake", description="Lake name"),
                           sunday: str = validate.path_date(description="The Sunday in YYYYmmdd format")):
    """
    NetCDF simulation results for a one-week period.

    Weeks are specified by providing the date (YYYYmmdd) for the Sunday proceeding the desired week.
    """
    validate.sunday(sunday)
    return simulations.get_simulations_file(filesystem, model, lake, sunday)


@app.get("/simulations/point/{model}/{lake}/{start_time}/{end_time}/{depth}/{lat}/{lng}", tags=["Simulations"])
async def simulations_point(model: simulations.Models = Path(..., title="Model", description="Model name"),
                            lake: simulations.Lakes = Path(..., title="Lake", description="Lake name"),
                            start_time: str = validate.path_time(description="The start time in YYYYmmddHHMM format (UTC)", example="202309050300"),
                            end_time: str = validate.path_time(description="The end time in YYYYmmddHHMM format (UTC)", example="202309072300"),
                            depth: float = validate.path_depth(),
                            lat: float = validate.path_latitude(),
                            lng: float = validate.path_longitude()):
    """
    Simulated timeseries of lake water temperature and velocity for a given location and depth.

    Outputs:
    - time: YYYYmmddHHMM
    - temperature: Water temperature (degC)
    - u:  Eastward flow velocity (m/s)
    - v: Northward flow velocity (m/s)
    - distance: Distance from requested location to center of closest grid point (m)
    - depth: Distance from the surface to the closest grid point to requested depth (m)
    """
    validate.time_range(start_time, end_time)
    return simulations.get_simulations_point(filesystem, model, lake, start_time, end_time, depth, lat, lng)


@app.get("/simulations/layer/{model}/{lake}/{time}/{depth}", tags=["Simulations"])
async def simulations_layer(model: simulations.Models = Path(..., title="Model", description="Model name"),
                            lake: simulations.Lakes = Path(..., title="Lake", description="Lake name"),
                            time: str = validate.path_time(),
                            depth: float = validate.path_depth()):
    """
    Simulated temperature and velocity for a depth layer at a specific time.
    """
    validate.time(time)
    return simulations.get_simulations_layer(filesystem, model, lake, time, depth)


@app.get("/simulations/layer_alplakes/{model}/{lake}/{parameter}/{start_time}/{end_time}/{depth}", tags=["Simulations"],
         response_class=PlainTextResponse)
async def simulations_layer_alplakes(model: simulations.Models = Path(..., title="Model", description="Model name"),
                                     lake: simulations.Lakes = Path(..., title="Lake", description="Lake name"),
                                     parameter: simulations.Parameters = Path(..., title="Parameter", description="Parameter"),
                                     start_time: str = validate.path_time(description="The start time in YYYYmmddHHMM format (UTC)", example="202309050300"),
                                     end_time: str = validate.path_time(description="The end time in YYYYmmddHHMM format (UTC)", example="202309072300"),
                                     depth: float = validate.path_depth()):
    """
    Simulated temperature and velocity for a depth layer over a specific time range.

    **Warning:** This endpoint is designed for supplying data to the Alplakes website. The output is **not** self-explanatory.
    """
    validate.time_range(start_time, end_time)
    return simulations.get_simulations_layer_alplakes(filesystem, model, lake, parameter, start_time, end_time, depth)


@app.get("/simulations/layer/average_temperature/{model}/{lake}/{start_time}/{end_time}/{depth}", tags=["Simulations"])
async def simulations_layer_average_temperature(model: simulations.Models = Path(..., title="Model", description="Model name"),
                                                lake: simulations.Lakes = Path(..., title="Lake", description="Lake name"),
                                                start_time: str = validate.path_time(description="The start time in YYYYmmddHHMM format (UTC)", example="202309050300"),
                                                end_time: str = validate.path_time(description="The end time in YYYYmmddHHMM format (UTC)", example="202309072300"),
                                                depth: float = validate.path_depth()):
    """
    Timeseries of geospatial average temperature at a given depth. Temperature is in °C.
    """
    validate.time_range(start_time, end_time)
    return simulations.get_simulations_layer_average_temperature(filesystem, model, lake, start_time, end_time, depth)


@app.get("/simulations/profile/{model}/{lake}/{time}/{lat}/{lng}", tags=["Simulations"])
async def simulations_profile(model: simulations.Models = Path(..., title="Model", description="Model name"),
                              lake: simulations.Lakes = Path(..., title="Lake", description="Lake name"),
                              time: str = validate.path_time(),
                              lat: float = validate.path_latitude(),
                              lng: float = validate.path_longitude()):
    """
    Vertical profile for a specific time and location.
    """
    validate.time(time)
    return simulations.get_simulations_profile(filesystem, model, lake, time, lat, lng)


@app.get("/simulations/transect/{model}/{lake}/{time}/{lats}/{lngs}", tags=["Simulations"])
async def simulations_transect(model: simulations.Models = Path(..., title="Model", description="Model name"),
                               lake: simulations.Lakes = Path(..., title="Lake", description="Lake name"),
                               time: str = validate.path_time(),
                               lats: str = Path(..., title="Lats", description="Comma separated list of latitudes (WGS84), minimum 2", example="46.37,46.54"),
                               lngs: str = Path(..., title="Lngs", description="Comma separated list of longitudes (WGS84), minimum 2", example="6.56,6.54")):
    """
    Lake transect at a specific time. Distance is in meters.
    """
    validate.latitude_list(lats)
    validate.longitude_list(lngs)
    return simulations.get_simulations_transect(filesystem, model, lake, time, lats, lngs)


@app.get("/simulations/transect/{model}/{lake}/{start_time}/{end_time}/{lats}/{lngs}", tags=["Simulations"])
async def simulations_transect_period(model: simulations.Models = Path(..., title="Model", description="Model name"),
                                      lake: simulations.Lakes = Path(..., title="Lake", description="Lake name"),
                                      start_time: str = validate.path_time( description="The start time in YYYYmmddHHMM format (UTC)", example="202309050300"),
                                      end_time: str = validate.path_time( description="The end time in YYYYmmddHHMM format (UTC)", example="202309072300"),
                                      lats: str = Path(..., title="Lats", description="Comma separated list of latitudes (WGS84), minimum 2", example="46.37,46.54"),
                                      lngs: str = Path(..., title="Lngs", description="Comma separated list of longitudes (WGS84), minimum 2", example="6.56,6.54")):
    """
    Lake transect over a specific period. Distance is in meters.
    """
    validate.latitude_list(lats)
    validate.longitude_list(lngs)
    validate.time_range(start_time, end_time)
    return simulations.get_simulations_transect_period(filesystem, model, lake, start_time, end_time, lats, lngs)


@app.get("/simulations/depthtime/{model}/{lake}/{start_time}/{end_time}/{lat}/{lng}", tags=["Simulations"])
async def simulations_depth_time(model: simulations.Models = Path(..., title="Model", description="Model name"),
                                 lake: simulations.Lakes = Path(..., title="Lake", description="Lake name"),
                                 start_time: str = validate.path_time(description="The start time in YYYYmmddHHMM format (UTC)", example="202309050300"),
                                 end_time: str = validate.path_time(description="The end time in YYYYmmddHHMM format (UTC)", example="202309072300"),
                                 lat: float = validate.path_latitude(),
                                 lng: float = validate.path_longitude()):
    """
    Vertical profile for a specific period and location.
    """
    validate.time_range(start_time, end_time)
    return simulations.get_simulations_depthtime(filesystem, model, lake, start_time, end_time, lat, lng)


@app.get("/remotesensing/metadata", tags=["Remote Sensing"])
async def remote_sensing_metadata():
    """
    Directory of remote sensing product types, organised by lake, satellite and parameter.
    """
    return remotesensing.get_metadata()


@app.get("/remotesensing/products/{lake}/{satellite}/{parameter}", tags=["Remote Sensing"])
async def remote_sensing_products(lake: simulations.Lakes = Path(..., title="Lake", description="Lake name"),
                                  satellite: remotesensing.Satellites = Path(..., title="Satellite", description="Satellite name"),
                                  parameter: str = Path(..., title="Parameter", description="Parameter", example="chla")):
    """
    Metadata for full time series of remote sensing products for a given lake, satellite and parameter.
    See /remotesensing/metadata for input options.
    """
    return remotesensing.get_lake_products(lake, satellite, parameter)
