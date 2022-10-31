from fastapi import FastAPI, Query
from fastapi.responses import RedirectResponse
from fastapi.middleware.gzip import GZipMiddleware

from app import meteoswiss
from app import bafu

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


@app.get("/meteoswiss/cosmo/reanalysis/{model}/{start_date}/{end_date}/{ll_lat}/{ll_lng}/{ur_lat}/{ur_lng}", tags=["Meteoswiss"])
async def meteoswiss_cosmo_reanalysis(model: meteoswiss.CosmoReanalysis, start_date: str, end_date: str, ll_lat: float,
                                      ll_lng: float, ur_lat: float, ur_lng: float,
                                      variables: list[str] = Query(default=["T_2M", "U", "V", "GLOB", "RELHUM_2M", "PMSL", "CLCT"])):
    """
    Weather data from MeteoSwiss COSMO reanalysis:
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
    meteoswiss.verify_cosmo_reanalysis(model, variables, start_date, end_date, ll_lat, ll_lng, ur_lat, ur_lng)
    return meteoswiss.get_cosmo_reanalysis(filesystem, model, variables, start_date, end_date, ll_lat, ll_lng, ur_lat, ur_lng)


@app.get("/meteoswiss/cosmo/forecast/{model}/{forecast_date}/{ll_lat}/{ll_lng}/{ur_lat}/{ur_lng}", tags=["Meteoswiss"])
async def meteoswiss_cosmo_forecast(model: meteoswiss.CosmoForecast, forecast_date: str,
                                    ll_lat: float, ll_lng: float, ur_lat: float, ur_lng: float,
                                    variables: list[str] = Query(default=["T_2M", "U", "V", "GLOB", "RELHUM_2M", "PMSL", "CLCT"])):
    """
    Weather data from MeteoSwiss COSMO forecasts:
    - **cosmo**: select a COSMO product
        - VNXQ94 (forecast): Cosmo-1e 33 hour ensemble forecast
        - VNXZ32 (forecast): Cosmo-2e 5 day ensemble forecast
    - **forecast_date**: date of forecast "YYYYMMDD"
    - **ll_lat**: latitude of lower left corner of bounding box (WGS 84)
    - **ll_lng**: longitude of lower left corner of bounding box (WGS 84)
    - **ur_lat**: latitude of upper right corner of bounding box (WGS 84)
    - **ur_lng**: longitude of upper right corner of bounding box (WGS 84)
    """
    meteoswiss.verify_cosmo_forecast(model, variables, forecast_date, ll_lat, ll_lng, ur_lat, ur_lng)
    return meteoswiss.get_cosmo_forecast(filesystem, model, variables, forecast_date, ll_lat, ll_lng, ur_lat, ur_lng)


@app.get("/bafu/hydrodata/metadata", tags=["Bafu"])
async def bafu_hydrodata_metadata():
    """
    Geojson of all the available Bafu hydrodata.
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


@app.get("/bafu/hydrodata/predicted/{status}/{station_id}/{parameter}", tags=["Bafu"])
async def bafu_hydrodata_predicted(status: bafu.HydrodataPredicted, station_id: int, parameter: str):
    """
    Predicted hydrodata from Bafu:
    - **status**:
        - official: pqprevi-official
        - unofficial: pqprevi-unofficial
    - **station_id**: 4 digit station identification code
    - **parameter**: parameter to retrieve (get list from /bafu/hydrodata/metadata)
    """
    bafu.verify_hydrodata_predicted(status, station_id, parameter)
    return bafu.get_hydrodata_predicted(filesystem, status, station_id, parameter)


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
