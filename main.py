import os
import socket
from pathlib import Path
from config.config import settings
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import available_dates, by_date, by_forecast, heatmap, heatmap_series, latest_date, time_series


app = FastAPI(
    title="Fire Front Radar API",
    version="1.0.0",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Extent-3857", "X-Scale-Min", "X-Scale-Max"],
)


@app.get("/")
def root():
    return {"message": "FastAPI is working"}


API = settings.API_PREFIX
app.include_router(available_dates.router, prefix=API)
app.include_router(by_date.router, prefix=API)
app.include_router(by_forecast.router, prefix=API)
app.include_router(heatmap.router, prefix=API)
app.include_router(heatmap_series.router, prefix=API)
app.include_router(latest_date.router, prefix=API)
app.include_router(time_series.router, prefix=API)

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8090, reload=True, log_level="info")