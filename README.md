# 📁 Backend structure

backend/  
├── main.py              # FastAPI entry point  
├── zarr_loader.py       # Functions to load/process NetCDF/Zarr  
├── api/  
│   └── routes.py        # Define API routes  
├── data/  
│   └── nc/             # .nc files stored here  
│   └── zarr/           # .zarr files produced  
├── config.py            # Settings for paths, chunking, etc. (to be done) 



## 🔌 Basic API endpoints

| Endpoint      | Method | Description                                                    |
| ------------- | ------ | -------------------------------------------------------------- |
| `/metadata`   | GET    | Returns dimensions and coordinate ranges                       |
| `/tiles`      | GET    | Returns a tile (chunk) for a given `time`, `lat`, `lon` window |
| `/timeseries` | GET    | Returns time series for a given point (lat, lon)               |
| `/heatmap`    | GET    | Aggregated 2D array for a given time index                     |
