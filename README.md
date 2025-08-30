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


## Docker deployment
You need to have docker installed in your machine. Docker Desktop for windows should be enough for a Windows PC. 

From inside the api directory use the command bellow to build the image:

```aiignore
docker buildx build --platform linux/amd64 -t cfe-backend --load .
```

The above command works when you build the image on an ARM machine (Mac with M1,M2,M3,M4 CPU or a Windows laptop with Snapdragon CPU). This might work on machines with Intel/AMD. If it doesn't try the command below.

```aiignore
docker build -t cfe-backend --load .
```

To run the app in a docker container you can use the command bellow:

```aiignore
docker run --name cfe-backend \
  -p 8090:8000 \
  -v "<data path on host>:/data:ro" \
  cfe-backend
```

Be careful! The string after `-t` (`cfe-backend` in the example) in build command should be the same with the one after `--name` in the run command as this is the name of the image.

If everything worked well then you should be able to access the fastAPI app on http://0.0.0.0:8090