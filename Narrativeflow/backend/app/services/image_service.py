"""
Image Generation Service - Local Stable Diffusion integration
Uses Automatic1111 WebUI API (default: http://localhost:7860)
"""
import httpx
import base64
import logging
from typing import Optional, Dict, Any
from pathlib import Path
import uuid
import os

from app.config import settings

logger = logging.getLogger(__name__)


class ImageGenerationService:
    """
    Service for local image generation using Stable Diffusion via Automatic1111 WebUI.
    
    Prerequisites:
    1. Install Stable Diffusion WebUI: https://github.com/AUTOMATIC1111/stable-diffusion-webui
    2. Run with API enabled: python launch.py --api
    3. Default runs on http://localhost:7860
    
    Alternatively, can use ComfyUI or other SD UIs with API support.
    """
    
    def __init__(self):
        # Configurable base URL for Stable Diffusion WebUI
        self.base_url = getattr(settings, 'sd_base_url', 'http://localhost:7860')
        self.output_dir = Path('static/generated_images')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.client = httpx.AsyncClient(timeout=300.0)  # 5 min timeout for image generation
        
        # Default generation settings
        self.default_settings = {
            "steps": 30,
            "cfg_scale": 7.5,
            "width": 512,
            "height": 512,
            "sampler_name": "DPM++ 2M Karras",
            "negative_prompt": "blurry, bad quality, distorted, ugly, deformed, nsfw, watermark, text, logo"
        }
        
        logger.info(f"Image Generation Service configured with SD WebUI at {self.base_url}")
    
    async def check_availability(self) -> Dict[str, Any]:
        """Check if Stable Diffusion WebUI is available and running."""
        try:
            response = await self.client.get(f"{self.base_url}/sdapi/v1/sd-models", timeout=5.0)
            if response.status_code == 200:
                models = response.json()
                return {
                    "available": True,
                    "models": [m.get("model_name", m.get("title", "Unknown")) for m in models[:5]],
                    "url": self.base_url
                }
        except Exception as e:
            logger.warning(f"Stable Diffusion WebUI not available: {e}")
        
        return {
            "available": False,
            "error": "Stable Diffusion WebUI is not running",
            "setup_instructions": (
                "To enable local image generation:\n"
                "1. Install Stable Diffusion WebUI from https://github.com/AUTOMATIC1111/stable-diffusion-webui\n"
                "2. Run with: python launch.py --api\n"
                "3. It should start on http://localhost:7860\n"
                "4. Download a model from https://civitai.com/ and place in models/Stable-diffusion/"
            )
        }
    
    async def generate_image(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 512,
        height: int = 512,
        steps: int = 30,
        cfg_scale: float = 7.5,
        seed: int = -1,
        style_preset: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate an image using Stable Diffusion.
        
        Args:
            prompt: The image generation prompt
            negative_prompt: What to avoid in the image
            width: Image width (default 512)
            height: Image height (default 512)
            steps: Number of generation steps (more = better quality but slower)
            cfg_scale: How closely to follow the prompt (7-10 recommended)
            seed: Random seed (-1 for random)
            style_preset: Optional style to apply (anime, photorealistic, fantasy, etc.)
        
        Returns:
            Dict with success status, image path or base64, and metadata
        """
        # Check if SD is available
        availability = await self.check_availability()
        if not availability.get("available"):
            return {
                "success": False,
                "error": "Stable Diffusion is not available",
                "details": availability
            }
        
        # Apply style presets
        enhanced_prompt = self._apply_style_preset(prompt, style_preset)
        final_negative = negative_prompt or self.default_settings["negative_prompt"]
        
        # Prepare the API payload
        payload = {
            "prompt": enhanced_prompt,
            "negative_prompt": final_negative,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "width": width,
            "height": height,
            "seed": seed,
            "sampler_name": self.default_settings["sampler_name"],
            "batch_size": 1,
            "n_iter": 1
        }
        
        try:
            logger.info(f"Generating image with prompt: {prompt[:100]}...")
            
            response = await self.client.post(
                f"{self.base_url}/sdapi/v1/txt2img",
                json=payload,
                timeout=300.0
            )
            
            if response.status_code != 200:
                return {
                    "success": False,
                    "error": f"SD API returned status {response.status_code}",
                    "details": response.text
                }
            
            result = response.json()
            
            if "images" not in result or not result["images"]:
                return {
                    "success": False,
                    "error": "No images returned from SD"
                }
            
            # Save the image
            image_data = result["images"][0]
            image_filename = f"{uuid.uuid4()}.png"
            image_path = self.output_dir / image_filename
            
            # Decode and save
            with open(image_path, "wb") as f:
                f.write(base64.b64decode(image_data))
            
            # Get the used seed from info
            info = result.get("info", "{}")
            if isinstance(info, str):
                import json
                try:
                    info = json.loads(info)
                except:
                    info = {}
            
            actual_seed = info.get("seed", seed)
            
            logger.info(f"Image generated successfully: {image_path}")
            
            return {
                "success": True,
                "image_path": f"/static/generated_images/{image_filename}",
                "image_base64": image_data,
                "prompt_used": enhanced_prompt,
                "negative_prompt": final_negative,
                "seed": actual_seed,
                "width": width,
                "height": height,
                "steps": steps,
                "cfg_scale": cfg_scale
            }
            
        except httpx.TimeoutException:
            return {
                "success": False,
                "error": "Image generation timed out. Try reducing steps or image size."
            }
        except Exception as e:
            logger.error(f"Image generation error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _apply_style_preset(self, prompt: str, style: Optional[str]) -> str:
        """Apply style presets to enhance the prompt."""
        style_additions = {
            "anime": "anime style, detailed anime art, vibrant colors, Studio Ghibli inspired",
            "photorealistic": "photorealistic, highly detailed, 8k uhd, professional photography, realistic lighting",
            "fantasy": "fantasy art style, magical atmosphere, ethereal lighting, detailed fantasy illustration",
            "dark": "dark fantasy, moody lighting, dramatic shadows, gothic atmosphere",
            "watercolor": "watercolor painting style, soft colors, artistic, traditional media look",
            "oil_painting": "oil painting style, classical art, rich colors, visible brushstrokes",
            "comic": "comic book style, bold lines, dynamic composition, superhero aesthetic",
            "portrait": "portrait photography, professional headshot, studio lighting, bokeh background",
            "landscape": "landscape photography, scenic view, golden hour lighting, breathtaking vista",
            "cyberpunk": "cyberpunk style, neon lights, futuristic city, high tech low life aesthetic",
            "steampunk": "steampunk aesthetic, brass and copper, victorian era technology, industrial design",
            "minimalist": "minimalist style, clean composition, simple shapes, elegant design"
        }
        
        if style and style.lower() in style_additions:
            return f"{prompt}, {style_additions[style.lower()]}"
        
        return prompt
    
    async def get_available_models(self) -> Dict[str, Any]:
        """Get list of available SD models."""
        try:
            response = await self.client.get(f"{self.base_url}/sdapi/v1/sd-models", timeout=10.0)
            if response.status_code == 200:
                models = response.json()
                return {
                    "success": True,
                    "models": [
                        {
                            "name": m.get("model_name", m.get("title", "Unknown")),
                            "hash": m.get("hash", ""),
                            "filename": m.get("filename", "")
                        }
                        for m in models
                    ]
                }
        except Exception as e:
            logger.error(f"Failed to get SD models: {e}")
        
        return {"success": False, "models": [], "error": "Could not retrieve models"}
    
    async def get_available_samplers(self) -> Dict[str, Any]:
        """Get list of available samplers."""
        try:
            response = await self.client.get(f"{self.base_url}/sdapi/v1/samplers", timeout=10.0)
            if response.status_code == 200:
                samplers = response.json()
                return {
                    "success": True,
                    "samplers": [s.get("name") for s in samplers]
                }
        except Exception as e:
            logger.error(f"Failed to get samplers: {e}")
        
        return {"success": False, "samplers": [], "error": "Could not retrieve samplers"}
    
    def build_character_prompt(
        self,
        character_name: str,
        physical_description: Optional[str] = None,
        distinguishing_features: Optional[list] = None,
        age: Optional[str] = None,
        gender: Optional[str] = None,
        occupation: Optional[str] = None,
        scene_context: Optional[str] = None,
        style: Optional[str] = None,
        custom_additions: Optional[str] = None
    ) -> str:
        """
        Build a consistent prompt for character image generation.
        Uses the character's stored attributes to maintain visual consistency.
        
        Args:
            character_name: The character's name
            physical_description: Text description of appearance
            distinguishing_features: List of notable features
            age: Character's age/age range
            gender: Character's gender
            occupation: Character's job/role
            scene_context: What the character is doing/where they are
            style: Art style preset
            custom_additions: Any additional prompt text
        
        Returns:
            A detailed prompt string for image generation
        """
        prompt_parts = []
        
        # Start with a portrait/character focus
        prompt_parts.append(f"portrait of {character_name}")
        
        # Add gender and age
        if gender:
            prompt_parts.append(gender)
        if age:
            prompt_parts.append(f"{age} years old" if age.isdigit() else age)
        
        # Add occupation if relevant
        if occupation:
            prompt_parts.append(f"{occupation}")
        
        # Add physical description
        if physical_description:
            # Clean and truncate if too long
            desc = physical_description.strip()
            if len(desc) > 300:
                desc = desc[:300] + "..."
            prompt_parts.append(desc)
        
        # Add distinguishing features
        if distinguishing_features:
            features = ", ".join(distinguishing_features[:5])  # Limit to 5 features
            prompt_parts.append(features)
        
        # Add scene context
        if scene_context:
            prompt_parts.append(scene_context)
        
        # Add custom additions
        if custom_additions:
            prompt_parts.append(custom_additions)
        
        # Join all parts
        base_prompt = ", ".join(prompt_parts)
        
        # Apply style if provided
        if style:
            base_prompt = self._apply_style_preset(base_prompt, style)
        
        # Add quality boosters
        base_prompt += ", high quality, detailed, masterpiece"
        
        return base_prompt
    
    def build_scene_prompt(
        self,
        scene_description: str,
        characters: Optional[list] = None,
        setting: Optional[str] = None,
        mood: Optional[str] = None,
        time_of_day: Optional[str] = None,
        style: Optional[str] = None
    ) -> str:
        """
        Build a prompt for scene/illustration generation.
        
        Args:
            scene_description: What's happening in the scene
            characters: List of character dicts with name and brief description
            setting: Where the scene takes place
            mood: Emotional tone of the scene
            time_of_day: Lighting hint (dawn, day, dusk, night)
            style: Art style preset
        
        Returns:
            A detailed prompt string for scene generation
        """
        prompt_parts = []
        
        # Main scene
        prompt_parts.append(scene_description)
        
        # Add characters
        if characters:
            char_descs = []
            for char in characters[:3]:  # Limit to 3 characters for clarity
                char_desc = char.get("name", "person")
                if char.get("brief_description"):
                    char_desc += f" ({char['brief_description']})"
                char_descs.append(char_desc)
            if char_descs:
                prompt_parts.append("featuring " + " and ".join(char_descs))
        
        # Add setting
        if setting:
            prompt_parts.append(f"in {setting}")
        
        # Add mood
        if mood:
            prompt_parts.append(f"{mood} atmosphere")
        
        # Add time of day for lighting
        time_lighting = {
            "dawn": "soft golden morning light, sunrise",
            "day": "bright daylight, clear lighting",
            "dusk": "warm sunset lighting, orange sky",
            "night": "moonlit, dark atmosphere, night scene"
        }
        if time_of_day and time_of_day.lower() in time_lighting:
            prompt_parts.append(time_lighting[time_of_day.lower()])
        
        # Join
        base_prompt = ", ".join(prompt_parts)
        
        # Apply style
        if style:
            base_prompt = self._apply_style_preset(base_prompt, style)
        
        # Add quality boosters
        base_prompt += ", high quality, detailed illustration, masterpiece"
        
        return base_prompt
    
    def get_consistency_tips(self) -> Dict[str, Any]:
        """
        Return tips for maintaining character consistency across images.
        """
        return {
            "tips": [
                "Use the same seed number for a character to maintain their base appearance",
                "Store detailed physical descriptions in the character profile",
                "Keep the same style preset for all images of a character",
                "Include distinguishing features (scars, hair color, eye color) in every prompt",
                "Use portrait_prompt field to save a working prompt that generates good results",
                "Consider using the same CFG scale and steps for consistency"
            ],
            "recommended_settings": {
                "character_portrait": {
                    "width": 512,
                    "height": 768,
                    "steps": 30,
                    "cfg_scale": 7.5,
                    "style": "portrait"
                },
                "scene_illustration": {
                    "width": 768,
                    "height": 512,
                    "steps": 35,
                    "cfg_scale": 7.0,
                    "style": "fantasy"
                },
                "book_cover": {
                    "width": 512,
                    "height": 768,
                    "steps": 40,
                    "cfg_scale": 8.0,
                    "style": "fantasy"
                }
            }
        }


# Singleton instance
image_service = ImageGenerationService()
