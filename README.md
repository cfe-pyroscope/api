# 📁 Backend structure

app/       
│     
├── api/       
│   ├── routes/    
│   │   └── available_dates.py  ← /api/{index}/available-dates    
│   │   └── by_dates.py         ← /api/{index}/by-date    
│   │   └── by_forecast.py      ← /api/{index}/by-forecast    
│   │   └── heatmap.py          ← /api/{index}/heatmap/image     
│   │   └── latest_date.py      ← /api/{index}/latest-date        
│   │   ├── time_series.py      ← /api/{index}/time-series     
│   │   └── tooltip.py          ← /api/{index}/tooltip      
│   └── __init__.py      
│      
├── config/  
│   ├── config.py               ← CORS, paths, constants     
│   └── logging_config.py       ← Central logger setup    
│    
├── data/  
│   ├── nc                      ← netCDF files    
│   └── zarr                    ← zarr files    
│   
├── misc/     
│   ├── create_zarr_file_v2.py  ← Merge netCDF files into a unique zarr   
│    
├── utils/     
│   ├── bounds_utils.py         ← BBox normalization and longitude logic       
│   └── heatmap_generator.py    ← Image generation and reprojection     
│   └── stats.py                ← Calculate statistics as mean and median        
│   ├── time_utils.py           ← Time indexing utilities for forecast datasets    
│   ├── zarr_loader.py          ← Data access and caching (Zarr/NetCDF)     
│      
├── main.py                     ← FastAPI app setup and router inclusion     
   