import os
from google import genai  # Import the Gemini API library
import tempfile
from app.core.logging import logger
from datetime import datetime
from app.core.config import settings
from app.core.task_tracker import task_tracker
from pydub import AudioSegment
from typing import Tuple, List, Optional
import math
import re
import json
import asyncio

OUTPUT_FOLDER = os.path.abspath("video_analysis_output")
openai_model = settings.openai_model
gemini_model = settings.gemini_model
if gemini_model:
    from google import genai
    client = genai.Client(api_key=settings.GEMINI_API_KEY)
if openai_model:
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

MAX_CHUNK_SIZE = 24 * 1024 * 1024  # 24MB to stay safely under the 25MB limit
CHUNK_DURATION = 1.5 * 60 * 1000  # 10 minutes in milliseconds

# NSFW content detection patterns
NSFW_PATTERNS = [
    r'\b(?:sex|porn|xxx|adult|nude|naked|explicit|nsfw)\b',
    r'\b(?:masturbat(?:e|ion)|orgasm|erotic)\b',
    r'\b(?:breast|boob|tit|ass|penis|vagina|dick|cock|pussy)\b',
    r'\b(?:fuck|shit|bitch|cunt|whore|slut)\b',
    r'\b(?:strip(?:ping|per)|escort|prostitut(?:e|ion))\b',
    r'\b(?:hentai|rule34|onlyfans)\b'
]

async def process_audio(video_content: bytes, task_id: str = None) -> Tuple[List[dict], Optional[str]]:
    """
    Process audio from video content, handling large files by splitting into chunks.
    """
    temp_video_file = None
    temp_audio_file = None
    output_folder = "video_analysis_output"
    
    try:
        # Create output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            logger.info(f"Output folder created/confirmed: {os.path.abspath(output_folder)}")
        
        # Save video content to temporary file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            temp_file.write(video_content)
            temp_video_file = temp_file.name
        logger.info(f"Temporary video file created at: {temp_video_file}")
        
        if task_id:
            task_tracker.update_progress(task_id, "Video file saved", 10)
        
        # Load video and extract audio
        video = AudioSegment.from_file(temp_video_file)
        if task_id:
            task_tracker.update_progress(task_id, "Video loaded for audio extraction", 15)
        
        # Save extracted audio
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_filename = os.path.join(output_folder, f"extracted_audio_{timestamp}.wav")
        video.export(audio_filename, format="wav")
        logger.info(f"Saved extracted audio locally: {audio_filename}")
        
        if task_id:
            task_tracker.update_progress(task_id, "Audio extracted and saved", 25)
        
        # Get file info
        file_size = os.path.getsize(audio_filename)
        logger.info(f"Audio file size: {file_size} bytes")
        
        # Process audio in chunks if necessary
        if task_id:
            task_tracker.update_progress(task_id, "Starting audio transcription", 30)
        
        logger.info("Transcribing audio using OpenAI Whisper API...")
        
        # Calculate number of chunks needed
        audio_length = len(video)
        logger.info(f"Audio length: {audio_length} ms")
        num_chunks = math.ceil(audio_length / CHUNK_DURATION)
        logger.info(f"Splitting audio into {num_chunks} chunks, ~{CHUNK_DURATION / 1000:.2f} seconds per chunk")        
        
        # Process audio in chunks
        transcriptions = []
        chunk_files = []
        
        try:
            for i in range(num_chunks):
                start_time = i * CHUNK_DURATION
                end_time = min((i + 1) * CHUNK_DURATION, audio_length)
                
                # Extract chunk
                chunk = video[start_time:end_time]
                
                # Save chunk to temporary file
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_chunk:
                    chunk.export(temp_chunk.name, format="wav")
                    chunk_files.append(temp_chunk.name)
                    
                    # Check chunk size
                    chunk_size = os.path.getsize(temp_chunk.name)
                    logger.info(f"Chunk {i+1}/{num_chunks} size: {chunk_size} bytes")
                    
                    if chunk_size > MAX_CHUNK_SIZE:
                        raise ValueError(f"Chunk {i+1} size ({chunk_size} bytes) exceeds maximum allowed size ({MAX_CHUNK_SIZE} bytes)")
                    
                    logger.info(f"file name {temp_chunk.name}")
                    with open(audio_filename, "rb") as audio_file:
                        try:
                            audio_data = audio_file.read()
                        except Exception as e:
                            logger.warning(f"Error reading audio file: {str(e)} {temp_chunk.name} {len(audio_file)}")
                    
                        try: 
                            assert len(audio_data) > 0
                        except Exception as e:
                            # logger.error(f"Chunk {i+1}/{num_chunks} is empty. Skipping transcription.")
                            logger.warning(f"Chunk {i+1}/{num_chunks} is empty. Skipping transcription. {len(audio_data)}")
                            transcriptions.append("")
                            continue

                        response = await client.aio.models.generate_content(
                            model='gemini-2.0-flash',
                            contents=[
                                "Transcribe the following audio file into text:",
                                genai.types.Part.from_bytes(data=audio_data,mime_type='audio/wav')
                                ],
                        )
                        
                        # Extract and return the transcription
                        transcriptions.append(response.text.strip())
                        
                    logger.info(f"Chunk {i+1}/{num_chunks} transcribed successfully")
                    
                    if task_id:
                        progress = 30 + (i + 1) * (35 - 30) / num_chunks
                        task_tracker.update_progress(task_id, f"Transcribed chunk {i+1}/{num_chunks}", progress)
            
            # Combine all transcriptions
            combined_text = " ".join(transcriptions)
            result = [{"text": combined_text}]
            logger.info(f"Audio Transcription: {result}")
            
            if task_id:
                task_tracker.update_progress(task_id, "Audio transcription completed", 35)
                task_tracker.update_progress(task_id, "Audio processing completed", 40)
            
            return result, audio_filename
            
        finally:
            # Clean up chunk files
            for chunk_file in chunk_files:
                if os.path.exists(chunk_file):
                    try:
                        os.unlink(chunk_file)
                        logger.info(f"Cleaned up chunk file: {chunk_file}")
                    except Exception as e:
                        logger.error(f"Error cleaning up chunk file: {str(e)}")
        
    except Exception as e:
        error_msg = f"Error in audio processing: {str(e)}"
        logger.error(error_msg)
        if task_id:
            task_tracker.update_progress(task_id, f"Error: {error_msg}", 35)
        return [{"error": error_msg}], None
        
    finally:
        # Clean up temporary files
        if temp_video_file and os.path.exists(temp_video_file):
            try:
                os.unlink(temp_video_file)
                logger.info(f"Cleaned up temporary video file: {temp_video_file}")
            except Exception as e:
                logger.error(f"Error cleaning up temporary video file: {str(e)}")

        if audio_filename and os.path.exists(audio_filename):
            try:
                os.unlink(audio_filename)
                logger.info(f"Cleaned up temporary audio file: {audio_filename}")
            except Exception as e:
                logger.error(f"Error cleaning up temporary audio file: {str(e)}")

async def check_content_safety(text: str) -> Tuple[bool, List[str]]:
    """
    Check if the content is safe by analyzing the text for NSFW content.
    Returns a tuple of (is_safe, warnings).
    """
    try:
        # First check for explicit patterns
        warnings = []
        for pattern in NSFW_PATTERNS:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                warnings.append(f"Detected inappropriate content: {match.group()}")
        
        if warnings:
            return False, warnings

        # Use OpenAI to check for more subtle NSFW content
        try:
            if openai_model:
                response = await client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {
                            "role": "system",
                            "content": """You are a very strict content moderator. Your task is to identify any inappropriate, 
                            adult, sexual, NSFW, or suggestive content in the text. Be extremely conservative - if there's any doubt,
                            mark it as inappropriate. Return a JSON object with:
                            {
                                "is_safe": boolean,
                                "warnings": [list of specific warnings],
                                "reason": "detailed explanation"
                            }"""
                        },
                        {
                            "role": "user",
                            "content": text
                        }
                    ],
                    temperature=0.1,
                    response_format={ "type": "json_object" },
                    timeout=30  # 30 seconds timeout
                )
                
                result = json.loads(response.choices[0].message.content)
            if gemini_model:
                prompt = f'''system: You are a very strict content moderator. Your task is to identify any inappropriate, 
                adult, sexual, NSFW, or suggestive content in the text. Be extremely conservative - if there's any doubt,
                mark it as inappropriate. Return a JSON object with:
                {{
                    "is_safe": boolean,
                    "warnings": [list of specific warnings],
                    "reason": "detailed explanation"
                }}
                user: {text}
                system:'''
                response = await client.aio.models.generate_content(model='gemini-2.0-flash',contents = prompt,  config=genai.types.GenerateContentConfig(temperature= 0.1, response_mime_type= 'application/json'))
                result = response.text.replace('```json','').replace('```','')

            if not result.get("is_safe", False):
                warnings.extend(result.get("warnings", []))
                reason = result.get("reason", "Content flagged as inappropriate")
                if reason and reason not in warnings:
                    warnings.append(reason)
                return False, warnings
            
            return True, []
            
        except asyncio.TimeoutError:
            logger.error("Timeout during content safety check")
            return False, ["Content safety check timed out"]
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing GPT response: {str(e)}")
            return False, ["Unable to verify content safety"]
        except Exception as e:
            logger.error(f"Error in GPT content check: {str(e)}")
            return False, ["Error checking content safety"]
        
    except Exception as e:
        logger.error(f"Error in content safety check: {str(e)}")
        return False, ["Error in content safety check"]