# 🎥 Video Description API

![Video Description API](https://img.shields.io/badge/API-Video%20Description-brightgreen)
![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.103%2B-blue)

This project provides a powerful API for analyzing videos, generating detailed descriptions, and extracting relevant keywords. It processes both video and audio content in parallel for efficient and comprehensive analysis.

## 📋 Table of Contents

1. [Features](#-features)
2. [Installation](#-installation)
3. [Usage](#-usage)
4. [API Endpoints](#-api-endpoints)
5. [Configuration](#-configuration)
6. [Contributing](#-contributing)

## 🌟 Features

- Parallel processing of video and audio content
- Extraction of key frames from videos
- Audio transcription using Google Gemini
- Detailed video description generation using gemini-2.0-flash
- Keyword extraction with relevance weighting
- Asynchronous processing with task ID for result retrieval
- Support for both file uploads and URL-based video processing

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Video-Description-using-API.git
   cd Video-Description-using-API
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up your environment variables:
   Create a `.env` file in the root directory and add your OpenAI API key:
   ```
   GEMINI_API_KEY = your_gemini_api_key_here
   ```

5. Run the FastAPI server:
   ```bash
   uvicorn main:app --reload
   ```

The API will be available at `http://localhost:8000`.

## 🎬 Usage

The API provides several endpoints for video analysis and result retrieval. Here's a brief overview of each endpoint:

## 🛠 API Endpoints

### POST /api/v1/analyze_video

Upload a video file or provide a URL to start the analysis process.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: 
  - `video`: The video file to analyze (optional)
  - `file_url`: URL of the video to analyze (optional)

**Response:**
```json
{
  "message": "Video analysis started.",
  "task_id": "unique_task_id"
}
```

### GET /api/v1/analysis_result/{task_id}

Retrieve the analysis results for a given task ID.

**Request:**
- Method: GET
- Path Parameter: 
  - `task_id`: The unique task ID returned from the analyze_video endpoint

**Response:**
```json
{
  "status": "completed",
  "description": "Detailed description of the video content",
  "keywords": [
    {
      "keyword": "example keyword",
      "weight": 10
    },
    ...
  ],
  "topics": ["topic1", "topic2", ...],
  "entities": ["entity1", "entity2", ...],
  "actions": ["action1", "action2", ...],
  "emotions": ["emotion1", "emotion2", ...],
  "visual_elements": ["element1", "element2", ...],
  "audio_elements": ["element1", "element2", ...],
  "genre": "video genre",
  "target_audience": ["audience1", "audience2", ...],
  "duration_estimate": "estimated duration",
  "quality_indicators": ["indicator1", "indicator2", ...],
  "unique_identifiers": ["identifier1", "identifier2", ...],
  "is_face_exist": true/false,
  "person_identity": "person name and gender",
  "other_person_identity": ["person1 name and gender", "person2 name and gender", ...],
  "psychological_personality": ["trait1", "trait2", "trait3"],
  "no_of_person_in_video": number,
  "transcript": "full transcript with timestamps"
}
```


## ⚙️ Configuration

The project uses environment variables for configuration. Create a `.env` file in the root directory with the following variables:

```
GEMINI_API_KEY = your_gemini_api_key_here
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
