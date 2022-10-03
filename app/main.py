from fastapi import FastAPI

from app import meteoswiss

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


@app.get("/")
def welcome():
    return {"Welcome to the Alplakes API from Eawag. Navigate to /docs or /redoc for documentation. For "
            "queries please contact James Runnalls (james.runnall@eawag.ch)."}


@app.get("/meteoswiss/cosmo/reanalysis/{model}/{start_time}/{end_time}/{ll_lat}/{ll_lng}/{ur_lat}/{ur_lng}", tags=["Meteoswiss"])
async def meteoswiss_cosmo_reanalysis(model: meteoswiss.CosmoReanalysis, start_time: int, end_time: int, ll_lat: float,
                                      ll_lng: float, ur_lat: float, ur_lng: float):
    """
    Weather data from MeteoSwiss COSMO reanalysis:
    - **cosmo**: select a COSMO product
        - VNXQ34 (reanalysis): Cosmo-1e 1 day deterministic
        - VNJK21 (reanalysis): Cosmo-1e 1 day ensemble forecast
    - **start_time**: unix time
    - **end_time**: unix time
    - **ll_lat**: Latitude of lower left corner of bounding box (WGS 84)
    - **ll_lng**: Longitude of lower left corner of bounding box (WGS 84)
    - **ur_lat**: Latitude of upper right corner of bounding box (WGS 84)
    - **ur_lng**: Longitude of upper right corner of bounding box (WGS 84)
    """
    meteoswiss.verify_cosmo_reanalysis(model, start_time, end_time, ll_lat, ll_lng, ur_lat, ur_lng)
    return meteoswiss.get_cosmo_reanalysis(model, start_time, end_time, ll_lat, ll_lng, ur_lat, ur_lng)


@app.get("/meteoswiss/cosmo/forecast/{model}/{forecast_date}/{ll_lat}/{ll_lng}/{ur_lat}/{ur_lng}", tags=["Meteoswiss"])
async def meteoswiss_cosmo_forecast(model: meteoswiss.CosmoForecast, forecast_date: str, ll_lat: float, ll_lng: float,
                                    ur_lat: float, ur_lng: float):
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
    meteoswiss.verify_cosmo_forecast(model, forecast_date, ll_lat, ll_lng, ur_lat, ur_lng)
    return meteoswiss.get_cosmo_forecast(model, forecast_date, ll_lat, ll_lng, ur_lat, ur_lng)

