
from fastapi import APIRouter, Query, Path, HTTPException
from fastapi.responses import JSONResponse

from config.config import settings
from config.logging_config import logger


router = APIRouter()

@router.get("/forecast_horizon")
async def forecast_horizon():
    try: 
        
        response = "hello"
        print("hello")
        logger.info(f"Time series response: {response}")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to build run-to-run series")
        return JSONResponse(status_code=400, content={"error": str(e)})
