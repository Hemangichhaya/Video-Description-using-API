from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import os

from app.api.routes import video_analysis
from app.core.config import settings
from app.core.logging import setup_logging
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title=settings.PROJECT_NAME, version=settings.PROJECT_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or restrict to ["http://localhost:5500"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    setup_logging()

# Serve your HTML file on "/"
@app.get("/", include_in_schema=False)
async def serve_frontend():
    return FileResponse("static/video.html")  # Path to your HTML

# Optional: Serve other static files (CSS, JS) if needed
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include your API router
app.include_router(video_analysis.router, prefix="/api/v1")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)