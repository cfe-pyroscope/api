# 📁 Backend structure

app/     
│     
├── api/     
│   ├── routes/   
│   │   └── available_dates.py      ← /api/available-dates  
│   │   └── latest_date.py      ← /api/latest-date    
│   │   ├── metadata.py         ← /api/{index}    
│   │   └── heatmap.py          ← /api/{index}/heatmap/image    
│   └── __init__.py    
│    
├── services/  
│   ├── zarr_loader.py          ← Data access and caching (Zarr/NetCDF)     
│   ├── bounds_utils.py         ← BBox normalization and longitude logic     
│   └── heatmap_generator.py    ← Image generation and reprojection   
│   ├── time_utils.py           ← Time indexing utilities for forecast datasets    
│    
├── main.py                     ← FastAPI app setup and router inclusion   
├── config.py                   ← CORS, paths, constants   
└── logging_config.py           ← Central logger setup   