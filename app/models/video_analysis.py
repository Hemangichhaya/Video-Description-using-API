from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

class VideoAnalysisResponse(BaseModel):
    description: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    audio_transcription: str
    whisper_info: Dict[str, Any] = Field(default_factory=dict)
    grid_image_path: Optional[str] = None
    audio_file_path: Optional[str] = None