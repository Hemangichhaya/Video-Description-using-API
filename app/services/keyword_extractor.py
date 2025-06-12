from app.core.config import settings
from app.core.logging import logger
import json

openai_model = settings.openai_model
gemini_model = settings.gemini_model
if gemini_model:
    from google import genai
    client = genai.Client(api_key=settings.GEMINI_API_KEY)
if openai_model:
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

async def extract_video_metadata(description: str, task_id: str = None, duration: str = None,is_safe: bool = None) -> dict:
    """
    Extract metadata from video description using GPT-4.
    
    Args:
        description (str): Video description
        task_id (str): Task identifier for progress tracking
        
    Returns:
        dict: Extracted metadata with all required fields
    """
    try:
        

        # Refined prompt for structured extraction
        prompt = f"""
        You are an expert content analyst. Your task is to analyze the given description and extract metadata in a clean, plain-text format. Follow these strict rules:

        1. **Avoid Special Characters**: Do not include any newline characters (`\n`), backslashes (`\\`), asterisks (`*`), underscores (`_`), or other Markdown syntax (e.g., bold `**text**`, italics `*text*`, or lists).
        2. **Plain Text Only**: Ensure the output is in plain text without any formatting or special symbols.
        3. **Valid JSON Structure**: Return the result in a valid JSON format, but ensure all values are plain text strings without unnecessary escape characters or formatting.
        4. **No Extra Line Breaks**: Remove all line breaks and ensure the text flows continuously in a single line where appropriate.
        5. **Focus on Content**: Prioritize the meaningful content and remove any decorative or unnecessary elements.

        generate keys based on given visual description and audio description. don't consider any context from the description.

        Video Description:
        {description}

        Extracted Metadata:
        {{
            "description": "{description}",  // Original video description
            "keywords": [
                {{"keyword":"string","weight":int}}  // Extract 10 most relevant keywords with weights (1-10) and make sure atleast 5 keywords are present
            ],
            "topics": ["string"],  // List at least 3 key topics discussed
            "entities": ["string"],  // Mentioned people, organizations, or objects
            "actions": ["string"],  // Key actions described
            "emotions": ["string"],  // Emotional tones present
            "visual_elements": ["string"],  // Notable visual elements
            "audio_elements": ["string"],  // Sound elements mentioned
            "genre": "string",  // Genre of the content
            "target_audience": ["string"],  // List of intended audiences
            "duration_estimate": "string",  // Estimated duration in minutes:seconds
            "quality_indicators": ["string"],  // Quality metrics or indicators
            "unique_identifiers": ["string"],  // Unique identifiers for the video
            "is_face_exist": bool,  // Whether faces are present in the video
            "person_identity": {{"name": "string", "gender": "string"}},  // Main person identity
            "other_person_identity": ["string"],  // Other persons' identities
            "psychological_personality": ["string"],  // Personality traits
            "no_of_person_in_video": int,  // Number of persons in the video if no person found then attach no_of_person_in_video = 0
            "content_warnings": ["string"],  // List of content warnings
            "safety_analysis": ["string"],  // Safety-related observations
            "is_safe": bool  // Whether the content is deemed safe
        }}
        
        Ensure all fields are filled based on the information available in the description. Return the response in valid JSON format with plain text only.
        """

        if openai_model:
            response = await client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert content analyzer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            # Parse and return the response
            extracted_metadata = json.loads(response.choices[0].message.content.strip())

        if gemini_model:
            response = await client.aio.models.generate_content(model='gemini-2.0-flash',contents = f'''system: You are an expert content analyzer., user: {prompt}, system:''',  config=genai.types.GenerateContentConfig(max_output_tokens= 1500, temperature=0.3, response_mime_type= 'application/json'))
            result = response.text.replace('```json', '').replace('```', '').strip()
  
            extracted_metadata = json.loads(result.strip())

        extracted_metadata["duration_estimate"] = duration
        if extracted_metadata["is_safe"] == True:
            extracted_metadata["is_safe"] = is_safe

        logger.info("Extracted metadata:")
        
        logger.info(json.dumps(extracted_metadata, indent=2))

        
        return extracted_metadata

    except Exception as e:
        logger.error(f"Error during metadata extraction: {str(e)}")
        return {}