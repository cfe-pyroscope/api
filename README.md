# 📁 Backend structure

app/     
│     
├── api/     
│   ├── routes/   
│   │   └── available_dates.py  ← /api/{index}/available-dates  
│   │   └── by_dates.py         ← /api/{index}/by-date  
│   │   └── by_forecast.py      ← /api/{index}/by-forecast   
│   │   └── latest_date.py      ← /api/{index}/latest-date      
│   │   ├── metadata.py         ← /api/{index}/metadata    
│   │   └── heatmap.py          ← /api/{index}/heatmap/image    
│   └── __init__.py    
│   
├── config/
│   ├── config.py               ← CORS, paths, constants   
│   └── logging_config.py       ← Central logger setup
│
├── utils/   
│   ├── bounds_utils.py         ← BBox normalization and longitude logic     
│   └── heatmap_generator.py    ← Image generation and reprojection   
│   ├── time_utils.py           ← Time indexing utilities for forecast datasets  
│   ├── zarr_loader.py          ← Data access and caching (Zarr/NetCDF)   
│    
├── main.py                     ← FastAPI app setup and router inclusion   
   