import base64
from google import genai
from app.core.task_tracker import task_tracker
from app.core.config import settings
from app.core.logging import logger
from typing import List
import re


openai_model = settings.openai_model
gemini_model = settings.gemini_model
if gemini_model:
    from google import genai
    client = genai.Client(api_key=settings.GEMINI_API_KEY)
if openai_model:
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

async def analyze_grid_images(base64_images: List[str], task_id: str = None) -> List[str]:
    """
    Analyze multiple grid images without audio and return their descriptions.
    
    Args:
        base64_images (List[str]): List of base64 encoded grid images
        task_id (str, optional): Task identifier for progress tracking
        
    Returns:
        List[str]: List of descriptions for each grid
    """
    try:
        descriptions = []
        total_images = len(base64_images)
        
        for idx, base64_image in enumerate(base64_images, 1):
            if task_id:
                progress = int(65 + (idx / total_images * 5))  # Progress from 65% to 70%
                task_tracker.update_progress(task_id, f"Analyzing grid image {idx}/{total_images}", progress)
                
            try:
                # Decode the base64 image
                image_data = base64.b64decode(base64_image)
            except Exception as e:
                logger.warning(f"Error decoding base64 image in grid {idx}: {str(e)}")
                continue
            prompt = """
                Analyze this series of video frames with particular attention to Christian themes and NSFW content:

                Provide a comprehensive description focusing on:
                1. The speaker's actions and expressions
                2. Any text overlays or icons and their significance
                3. Visual elements and their significance
                4. The overall theme and message visible in these frames
                5. Number of faces visible
                6. Gender identification of visible individuals
                7. Personality traits and demeanor of main individuals
                8. Notable interactions or expressions
                9. Visual progression and scene changes
                10. Any identifiable individuals or notable features

                Focus on visual analysis only. Describe the progression naturally without mentioning grid layout.
                Write the description only for 10 above points.
                """

            if openai_model:
                response = await client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens = 500
                )
                descriptions.append(response.choices[0].message.content.strip())

            if gemini_model:
                response = await client.aio.models.generate_content(
                                model='gemini-2.0-flash',
                                contents=[prompt,genai.types.Part.from_bytes(data=image_data, mime_type="image/png")],
                                config=genai.types.GenerateContentConfig(max_output_tokens= 400))
                descriptions.append(response.text.strip())
            
        return descriptions
    
    except Exception as e:
        error_msg = f"Error in analyzing grid images: {str(e)}"
        logger.error(error_msg)
        if task_id:
            task_tracker.update_progress(task_id, f"Error: {error_msg}", 70)
        return [error_msg]

async def generate_description(base64_images: List[str], audio_transcription: str = None, task_id: str = None) -> str:
    """
    Generate a comprehensive video description combining multiple grid analyses and audio transcription.
    
    Args:
        base64_images (List[str]): List of base64 encoded grid images
        audio_transcription (str, optional): Audio transcription text
        task_id (str, optional): Task identifier for progress tracking
        
    Returns:
        str: Combined comprehensive description
    """
    try:
        # First, analyze all grid images
        if task_id:
            task_tracker.update_progress(task_id, "Starting grid analysis", 65)
            
        grid_descriptions = await analyze_grid_images(base64_images, task_id)
        
        # Prepare the final analysis prompt
        if task_id:
            task_tracker.update_progress(task_id, "Generating final description", 70)
            
        combined_descriptions = "\n\n".join(grid_descriptions)
        
        final_prompt = f"""
        Based on the following video segment descriptions and audio transcription, provide a comprehensive analysis:

        Video Segments Analysis:
        {combined_descriptions}

        Audio Transcription:
        {audio_transcription if audio_transcription else "No audio transcription available."}

        Provide a unified description that includes:
        1. The speaker's actions and expressions
        2. Any text overlays or icons and their significance
        3. How the visuals complement or illustrate the audio content
        4. The overall theme and message of the video
        5. No of face visible in the video
        6. Count the number of persons in the video
        7. Assess the personality traits of the main individual featured, including insights into their demeanor and engagement
        8. Identify the gender of the main speaker at the beginning of the video, as well as the genders of all other individuals present
        9. If possible, provide names or identities of other individuals featured in the video
        10. Describe the speaker's actions, expressions, and any notable interactions
        11. Speaker Identification: If there are multiple speakers, indicate who is speaking at each point
        12. Use paragraph breaks for different speakers or topics to enhance readability

        Focus ONLY on video and audio-related content.
        Do not mention anything about a grid or layout of the images.
        Avoid prefacing the analysis with statements like "Okay, here is an analysis..." or similar introductory phrases.
        Provide a natural, flowing narrative that combines all these elements into a coherent analysis.
        """
        
        if openai_model:
            response = await client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "user",
                        "content": final_prompt
                    }
                ],
                max_tokens=1500
            )
            
            if task_id:
                task_tracker.update_progress(task_id, "Description generation completed", 75)
                
            return response.choices[0].message.content.strip()
        if gemini_model:
            response = await client.aio.models.generate_content(model='gemini-2.0-flash',contents = f'''user: {final_prompt}, system:''',  config=genai.types.GenerateContentConfig(max_output_tokens= 1500))
            result = response.text.replace('```json','').replace('```','')
    
            if task_id:
                task_tracker.update_progress(task_id, "Description generation completed", 75)
            
            return re.sub(r"[\n*\\]", " ", result.strip())
        
    except Exception as e:
        error_msg = f"Error in generate_description: {str(e)}"
        logger.error(error_msg)
        if task_id:
            task_tracker.update_progress(task_id, f"Error: {error_msg}", 75)
        return error_msg