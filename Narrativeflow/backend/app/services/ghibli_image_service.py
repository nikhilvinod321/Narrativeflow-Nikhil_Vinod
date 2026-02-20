"""
Local Image Generation Service
Optimized for AMD integrated graphics with DirectML
Uses SD-Turbo for fast generation with various art styles
"""
import asyncio
import base64
import logging
import uuid
from pathlib import Path
from typing import Optional, Dict, Any
from io import BytesIO
from PIL import Image
import os

logger = logging.getLogger(__name__)

# Global model holder to avoid reloading
_pipeline = None
_model_loaded = False

# Art style definitions
ART_STYLES = {
    "ghibli": {
        "name": "Studio Ghibli",
        "prompt": "studio ghibli style, ghibli anime, hayao miyazaki style, hand-drawn animation, soft watercolor, gentle lighting, whimsical, dreamy atmosphere, detailed backgrounds, warm colors, nostalgic feeling",
        "negative": "photorealistic, 3d render, cgi, dark, gritty, violent, horror",
    },
    "anime": {
        "name": "Anime/Manga",
        "prompt": "anime style, manga art, japanese animation, cel shading, vibrant colors, expressive eyes, detailed linework",
        "negative": "photorealistic, 3d render, western cartoon, sketch",
    },
    "photorealistic": {
        "name": "Photorealistic",
        "prompt": "photorealistic, highly detailed, 8k uhd, professional photography, natural lighting, sharp focus, realistic textures",
        "negative": "cartoon, anime, painting, drawing, illustration, sketch",
    },
    "fantasy": {
        "name": "Fantasy Art",
        "prompt": "fantasy art style, epic fantasy, magical atmosphere, detailed illustration, dramatic lighting, rich colors, concept art quality",
        "negative": "modern, mundane, photorealistic, minimalist",
    },
    "watercolor": {
        "name": "Watercolor",
        "prompt": "watercolor painting, soft brushstrokes, flowing colors, artistic, delicate details, paper texture, traditional art",
        "negative": "digital art, photorealistic, sharp edges, 3d render",
    },
    "oil_painting": {
        "name": "Oil Painting",
        "prompt": "oil painting style, classical art, rich colors, visible brushstrokes, museum quality, fine art, masterpiece",
        "negative": "digital art, photorealistic, cartoon, anime",
    },
    "comic": {
        "name": "Comic Book",
        "prompt": "comic book art style, bold lines, dynamic composition, superhero comics, graphic novel, inked artwork, halftone dots",
        "negative": "photorealistic, anime, watercolor, soft",
    },
    "cyberpunk": {
        "name": "Cyberpunk",
        "prompt": "cyberpunk style, neon lights, futuristic city, high tech low life, rain-slicked streets, holographic displays, dark atmosphere with neon accents",
        "negative": "fantasy, medieval, nature, bright daylight, pastoral",
    },
    "steampunk": {
        "name": "Steampunk",
        "prompt": "steampunk style, victorian era, brass and copper machinery, gears and cogs, steam-powered, goggles, airships, sepia tones",
        "negative": "modern, futuristic, digital, minimalist",
    },
    "dark_gothic": {
        "name": "Dark/Gothic",
        "prompt": "dark gothic art, moody atmosphere, dramatic shadows, mysterious, elegant darkness, baroque influence, haunting beauty",
        "negative": "bright, cheerful, colorful, cartoon, cute",
    },
    "minimalist": {
        "name": "Minimalist",
        "prompt": "minimalist art style, clean lines, simple shapes, limited color palette, negative space, modern design, elegant simplicity",
        "negative": "detailed, busy, cluttered, baroque, ornate",
    },
    "pixel_art": {
        "name": "Pixel Art",
        "prompt": "pixel art style, 16-bit graphics, retro game aesthetic, limited color palette, crisp pixels, nostalgic gaming",
        "negative": "smooth, high resolution, photorealistic, gradients",
    },
    "impressionist": {
        "name": "Impressionist",
        "prompt": "impressionist painting style, visible brushstrokes, light and color focus, monet style, outdoor scenes, atmospheric, soft edges",
        "negative": "sharp lines, digital art, photorealistic, dark",
    },
    "art_nouveau": {
        "name": "Art Nouveau",
        "prompt": "art nouveau style, flowing organic lines, decorative patterns, alphonse mucha inspired, elegant curves, nature motifs, ornamental",
        "negative": "minimalist, modern, geometric, harsh lines",
    },
}


class GhibliImageService:
    """
    Local image generation service optimized for low VRAM.
    Uses SD-Turbo with various art styles for fast, consistent results.
    """
    
    def __init__(self):
        self.output_dir = Path('static/generated_images')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_id = "stabilityai/sd-turbo"  # Fast model, fewer steps needed
        self.use_directml = True
        
        # Default to Ghibli style
        self.ghibli_style = ART_STYLES["ghibli"]["prompt"]
        
        # Default negative prompt
        self.negative_prompt = (
            "bad anatomy, bad proportions, blurry, low quality, "
            "watermark, text, logo, signature, nsfw"
        )
        
        # Recommended settings
        self.default_settings = {
            "steps": 4,  # SD-Turbo only needs 4 steps
            "guidance_scale": 0.0,  # SD-Turbo works with 0 guidance
            "width": 512,
            "height": 512,
        }
        
        logger.info("Local Image Service initialized with multiple style support")
    
    def get_available_styles(self) -> Dict[str, Any]:
        """Return all available art styles."""
        return {
            "styles": [
                {"id": style_id, "name": style_data["name"]}
                for style_id, style_data in ART_STYLES.items()
            ],
            "default": "ghibli"
        }
    
    def get_style_prompt(self, style_id: str) -> tuple:
        """Get the prompt and negative prompt for a style."""
        style = ART_STYLES.get(style_id, ART_STYLES["ghibli"])
        return style["prompt"], style.get("negative", self.negative_prompt)
    
    async def check_availability(self) -> Dict[str, Any]:
        """Check if the service is available and return system info."""
        try:
            import torch
            import onnxruntime as ort
            
            # Check DirectML availability
            providers = ort.get_available_providers()
            has_directml = 'DmlExecutionProvider' in providers
            
            return {
                "available": True,
                "backend": "DirectML" if has_directml else "CPU",
                "providers": providers,
                "model": self.model_id,
                "style": "Ghibli",
                "estimated_time": "1-3 minutes per image",
                "ram_available": True,
                "note": "Using SD-Turbo with DirectML acceleration for AMD GPU"
            }
        except ImportError as e:
            return {
                "available": False,
                "error": f"Missing dependency: {e}",
                "setup_instructions": "Run: pip install onnxruntime-directml diffusers transformers"
            }
        except Exception as e:
            return {
                "available": False,
                "error": str(e)
            }
    
    def _load_pipeline(self):
        """Load the image generation pipeline (lazy loading)."""
        global _pipeline, _model_loaded
        
        if _model_loaded and _pipeline is not None:
            return _pipeline
        
        logger.info("Loading SD-Turbo pipeline...")
        
        try:
            import torch
            from diffusers import DiffusionPipeline
            
            # Use DiffusionPipeline which auto-detects the correct pipeline class
            # This avoids import issues with specific pipeline classes
            _pipeline = DiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float32,
                use_safetensors=True,
            )
            
            # Move to CPU (DirectML integration is complex, CPU is more stable)
            _pipeline = _pipeline.to("cpu")
            
            # Enable memory optimizations if available
            if hasattr(_pipeline, 'enable_attention_slicing'):
                _pipeline.enable_attention_slicing()
            
            _model_loaded = True
            logger.info("SD-Turbo pipeline loaded successfully")
            return _pipeline
            
        except Exception as e:
            logger.error(f"Failed to load pipeline: {e}")
            raise
    
    def _extract_scene_description(self, text: str, max_words: int = 30) -> str:
        """
        Extract a concise visual scene description from story text.
        Focuses on setting, characters, actions, and atmosphere.
        """
        import re
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        # Remove special characters but keep some punctuation
        text = re.sub(r'[^\w\s,.\'-]', ' ', text)
        # Normalize whitespace
        text = ' '.join(text.split())
        
        # If text is short enough, return as-is
        words = text.split()
        if len(words) <= max_words:
            return text
        
        # Try to find the most visually descriptive sentence
        sentences = re.split(r'[.!?]+', text)
        
        # Keywords that indicate visual/scene descriptions
        visual_indicators = {
            'location': ['stood', 'sat', 'walked', 'room', 'building', 'outside', 'inside', 
                        'window', 'door', 'street', 'forest', 'sky', 'ocean', 'mountain'],
            'appearance': ['wore', 'wearing', 'hair', 'eyes', 'face', 'tall', 'young', 'old',
                          'beautiful', 'handsome', 'dressed'],
            'action': ['looked', 'stared', 'watched', 'gazed', 'ran', 'walked', 'stood',
                      'held', 'reached', 'touched'],
            'atmosphere': ['dark', 'bright', 'quiet', 'loud', 'peaceful', 'tense', 'cold',
                          'warm', 'misty', 'sunny', 'rainy', 'night', 'morning', 'evening']
        }
        
        # Score sentences by visual content
        best_sentence = ""
        best_score = 0
        
        for sentence in sentences[:10]:  # Only check first 10 sentences
            sentence = sentence.strip()
            if len(sentence) < 20:  # Skip very short sentences
                continue
                
            score = 0
            sentence_lower = sentence.lower()
            
            for category, keywords in visual_indicators.items():
                for kw in keywords:
                    if kw in sentence_lower:
                        score += 1
            
            if score > best_score:
                best_score = score
                best_sentence = sentence
        
        # If we found a good visual sentence, use it
        if best_sentence and best_score >= 2:
            words = best_sentence.split()[:max_words]
            result = ' '.join(words)
            logger.info(f"Extracted visual sentence (score {best_score}): {result[:50]}...")
            return result
        
        # Fallback: Build a description from key elements
        visual_elements = []
        
        # Extract character names (Dr. Voss, Alex, etc.)
        names = re.findall(r'\b(?:Dr\.|Mr\.|Mrs\.|Ms\.)?[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        unique_names = list(dict.fromkeys(names))[:2]  # Keep order, max 2
        if unique_names:
            visual_elements.append(' and '.join(unique_names))
        
        # Find location/setting words
        location_words = []
        location_patterns = [
            r'\b(control room|control panel|laboratory|office|city|forest|ocean|mountain|castle|village|room|building|tower|ship|space station)\b',
            r'\b(stood before|sat at|walked through|entered the|inside the|outside the)\s+(?:a\s+)?(\w+(?:\s+\w+)?)\b'
        ]
        for pattern in location_patterns:
            matches = re.findall(pattern, text.lower())
            if matches:
                if isinstance(matches[0], tuple):
                    location_words.extend([m[1] if len(m) > 1 else m[0] for m in matches])
                else:
                    location_words.extend(matches)
        
        if location_words:
            visual_elements.append(location_words[0])
        
        # Find action/pose
        action_patterns = [
            r'\b(stood|sat|walked|ran|looked|watched|worked|stared)\b',
        ]
        for pattern in action_patterns:
            match = re.search(pattern, text.lower())
            if match:
                visual_elements.append(match.group(1) + 'ing')
                break
        
        # Find atmosphere words
        atmosphere_words = ['dark', 'bright', 'quiet', 'peaceful', 'tense', 'mysterious', 
                          'warm', 'cold', 'eerie', 'calm', 'busy']
        for word in atmosphere_words:
            if word in text.lower():
                visual_elements.append(word + ' atmosphere')
                break
        
        if visual_elements:
            result = ', '.join(visual_elements)
            logger.info(f"Built visual elements: {result}")
            return result
        
        # Final fallback: first sentence truncated
        first_sentence = sentences[0] if sentences else text
        words = first_sentence.split()[:max_words]
        return ' '.join(words)
    
    def build_styled_prompt(
        self,
        subject: str,
        style_id: str = "ghibli",
        scene_context: Optional[str] = None,
        mood: Optional[str] = None,
        time_of_day: Optional[str] = None,
        additional_details: Optional[str] = None
    ) -> tuple:
        """
        Build an optimized prompt for any art style.
        
        IMPORTANT: CLIP can only handle 77 tokens, so we put style keywords FIRST.
        
        Returns:
            Tuple of (prompt, negative_prompt)
        """
        # Get style-specific prompts
        style_prompt, style_negative = self.get_style_prompt(style_id)
        
        # START with style (most important - put first so it's not truncated)
        prompt_parts = [style_prompt]
        
        # Add mood EARLY (important for style)
        mood_mappings = {
            "happy": "joyful, bright atmosphere",
            "sad": "melancholic, emotional",
            "peaceful": "serene, tranquil",
            "adventurous": "dynamic, sense of wonder",
            "mysterious": "ethereal, magical, misty",
            "romantic": "soft pink tones, romantic atmosphere",
            "dramatic": "dramatic lighting, emotional intensity",
            "dark": "dark atmosphere, moody",
            "epic": "epic scale, grand composition"
        }
        if mood and mood.lower() in mood_mappings:
            prompt_parts.append(mood_mappings[mood.lower()])
        elif mood:
            prompt_parts.append(mood)
        
        # Add time of day for lighting
        time_mappings = {
            "dawn": "golden morning light, sunrise",
            "day": "bright daylight, blue sky",
            "sunset": "warm sunset, golden hour",
            "dusk": "purple twilight, evening",
            "night": "moonlit, starry sky, night scene"
        }
        if time_of_day and time_of_day.lower() in time_mappings:
            prompt_parts.append(time_mappings[time_of_day.lower()])
        
        # Extract and add scene description from subject
        scene_description = self._extract_scene_description(subject, max_words=25)
        if scene_description:
            prompt_parts.append(scene_description)
        
        # Add scene context (brief)
        if scene_context:
            context_desc = self._extract_scene_description(scene_context, max_words=10)
            prompt_parts.append(context_desc)
        
        # Add extra details (brief)
        if additional_details:
            details_desc = self._extract_scene_description(additional_details, max_words=10)
            prompt_parts.append(details_desc)
        
        # Combine - style is FIRST, content is AFTER
        full_prompt = ", ".join(prompt_parts)
        
        # Final safety truncation
        max_chars = 300
        if len(full_prompt) > max_chars:
            full_prompt = full_prompt[:max_chars].rsplit(',', 1)[0]
        
        # Combine negative prompts
        full_negative = f"{self.negative_prompt}, {style_negative}"
        
        logger.info(f"Built {style_id} prompt ({len(full_prompt)} chars): {full_prompt[:100]}...")
        
        return full_prompt, full_negative
    
    def build_ghibli_prompt(
        self,
        subject: str,
        scene_context: Optional[str] = None,
        mood: Optional[str] = None,
        time_of_day: Optional[str] = None,
        additional_details: Optional[str] = None
    ) -> str:
        """
        Build a Ghibli-style optimized prompt.
        Wrapper for build_styled_prompt with ghibli style.
        """
        prompt, _ = self.build_styled_prompt(
            subject=subject,
            style_id="ghibli",
            scene_context=scene_context,
            mood=mood,
            time_of_day=time_of_day,
            additional_details=additional_details
        )
        return prompt
    
    def build_character_prompt(
        self,
        name: str,
        physical_description: Optional[str] = None,
        age: Optional[str] = None,
        gender: Optional[str] = None,
        distinguishing_features: Optional[list] = None,
        expression: Optional[str] = None,
        pose: Optional[str] = None
    ) -> str:
        """
        Build a Ghibli-style character portrait prompt.
        Puts style keywords FIRST to avoid CLIP truncation.
        """
        # Start with Ghibli style and character-specific keywords (FIRST to avoid truncation)
        parts = [
            "studio ghibli style character portrait",
            "ghibli anime, hayao miyazaki style",
            "expressive eyes, soft features, hand-drawn animation"
        ]
        
        # Expression (important for portrait)
        if expression:
            parts.append(f"{expression} expression")
        else:
            parts.append("gentle expression")
        
        # Character basics
        char_desc = ""
        if gender:
            char_desc = f"{gender}"
        if age:
            if str(age).isdigit():
                char_desc += f" {age} years old" if char_desc else f"{age} years old"
            else:
                char_desc += f" {age}" if char_desc else age
        if char_desc:
            parts.append(char_desc.strip())
        
        # Physical description (extract key visual traits)
        if physical_description:
            # Extract key visual words from description
            desc_keywords = self._extract_scene_description(physical_description, max_words=15)
            if desc_keywords:
                parts.append(desc_keywords)
        
        # Distinguishing features
        if distinguishing_features:
            features = ", ".join(str(f) for f in distinguishing_features[:3])
            parts.append(features)
        
        # Pose
        if pose:
            parts.append(pose)
        
        # Combine and truncate
        full_prompt = ", ".join(parts)
        
        # Safety truncation for CLIP (77 tokens max)
        max_chars = 300
        if len(full_prompt) > max_chars:
            full_prompt = full_prompt[:max_chars].rsplit(',', 1)[0]
        
        logger.info(f"Built character prompt ({len(full_prompt)} chars): {full_prompt[:100]}...")
        
        return full_prompt
    
    async def generate_image(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 512,
        height: int = 512,
        seed: int = -1,
        steps: int = 4,
        guidance_scale: float = 0.0
    ) -> Dict[str, Any]:
        """
        Generate a Ghibli-style image.
        
        SD-Turbo is optimized for 4 steps with guidance_scale=0.
        """
        try:
            import torch
            
            # Load pipeline if not loaded
            pipeline = self._load_pipeline()
            
            # Set seed for reproducibility
            generator = None
            actual_seed = seed
            if seed == -1:
                actual_seed = torch.randint(0, 2**32 - 1, (1,)).item()
            generator = torch.Generator("cpu").manual_seed(actual_seed)
            
            # Use default negative prompt if not provided
            neg_prompt = negative_prompt or self.negative_prompt
            
            logger.info(f"Generating image with seed {actual_seed}...")
            
            # Run generation in thread pool to not block
            def generate():
                return pipeline(
                    prompt=prompt,
                    negative_prompt=neg_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                ).images[0]
            
            # Run in executor to not block async loop
            loop = asyncio.get_event_loop()
            image = await loop.run_in_executor(None, generate)
            
            # Save image
            image_filename = f"ghibli_{uuid.uuid4()}.png"
            image_path = self.output_dir / image_filename
            image.save(image_path, "PNG")
            
            # Convert to base64
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            logger.info(f"Image generated successfully: {image_path}")
            
            return {
                "success": True,
                "image_path": f"/static/generated_images/{image_filename}",
                "image_base64": image_base64,
                "seed": actual_seed,
                "prompt": prompt,
                "negative_prompt": neg_prompt,
                "width": width,
                "height": height,
                "steps": steps,
                "style": "ghibli"
            }
            
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "prompt": prompt
            }
    
    async def generate_character_portrait(
        self,
        name: str,
        physical_description: Optional[str] = None,
        age: Optional[str] = None,
        gender: Optional[str] = None,
        distinguishing_features: Optional[list] = None,
        expression: Optional[str] = None,
        seed: int = -1
    ) -> Dict[str, Any]:
        """Generate a Ghibli-style character portrait."""
        prompt = self.build_character_prompt(
            name=name,
            physical_description=physical_description,
            age=age,
            gender=gender,
            distinguishing_features=distinguishing_features,
            expression=expression
        )
        
        # Use portrait dimensions (taller)
        return await self.generate_image(
            prompt=prompt,
            width=512,
            height=640,
            seed=seed
        )
    
    async def generate_scene(
        self,
        description: str,
        characters: Optional[list] = None,
        mood: Optional[str] = None,
        time_of_day: Optional[str] = None,
        seed: int = -1,
        style_id: str = "ghibli"
    ) -> Dict[str, Any]:
        """Generate a styled scene illustration."""
        subject_parts = [description]
        
        # Add characters
        if characters:
            char_names = [c.get("name", "character") for c in characters[:2]]
            if char_names:
                subject_parts.append(f"featuring {' and '.join(char_names)}")
        
        prompt, negative = self.build_styled_prompt(
            subject=", ".join(subject_parts),
            mood=mood,
            time_of_day=time_of_day,
            style_id=style_id
        )
        
        # Use landscape dimensions
        return await self.generate_image(
            prompt=prompt,
            negative_prompt=negative,
            width=768,
            height=512,
            seed=seed
        )
    
    async def generate_styled_character(
        self,
        name: str,
        physical_description: str,
        personality: Optional[str] = None,
        expression: Optional[str] = None,
        background: Optional[str] = None,
        seed: int = -1,
        style_id: str = "ghibli"
    ) -> Dict[str, Any]:
        """Generate a styled character portrait."""
        prompt, negative = self.build_styled_prompt(
            subject=f"portrait of {name}, {physical_description}",
            mood=personality,
            style_id=style_id
        )
        
        # Add expression if provided
        if expression:
            prompt = prompt.replace(name, f"{name} with {expression} expression")
        
        # Add background if provided
        if background:
            prompt += f", {background} background"
        
        # Use portrait dimensions
        return await self.generate_image(
            prompt=prompt,
            negative_prompt=negative,
            width=512,
            height=640,
            seed=seed
        )
    
    def get_style_presets(self) -> Dict[str, Any]:
        """Get available style presets including all art styles."""
        return {
            "art_styles": [
                {"id": style_id, "label": info["name"]} 
                for style_id, info in ART_STYLES.items()
            ],
            "moods": [
                {"id": "peaceful", "label": "Peaceful & Serene"},
                {"id": "adventurous", "label": "Adventurous"},
                {"id": "mysterious", "label": "Mysterious & Magical"},
                {"id": "romantic", "label": "Romantic"},
                {"id": "melancholic", "label": "Melancholic"},
                {"id": "joyful", "label": "Joyful & Playful"},
            ],
            "times_of_day": [
                {"id": "dawn", "label": "Dawn - Soft Morning Light"},
                {"id": "day", "label": "Day - Bright & Cheerful"},
                {"id": "sunset", "label": "Sunset - Golden Hour"},
                {"id": "dusk", "label": "Dusk - Twilight Magic"},
                {"id": "night", "label": "Night - Moonlit Scene"},
            ],
            "character_expressions": [
                {"id": "gentle", "label": "Gentle & Kind"},
                {"id": "determined", "label": "Determined"},
                {"id": "curious", "label": "Curious & Wonder"},
                {"id": "peaceful", "label": "Peaceful"},
                {"id": "joyful", "label": "Joyful"},
                {"id": "thoughtful", "label": "Thoughtful"},
            ],
            "tips": [
                "Different art styles work better with different subjects",
                "Photorealistic style needs detailed descriptions",
                "Anime/Ghibli styles work best with expressive characters",
                "Fantasy and Cyberpunk styles excel at dramatic scenes",
                "Watercolor and Impressionist work great for landscapes",
                "Save the seed after getting a good result for consistency",
                "Portrait dimensions (512x640) work best for characters",
                "Landscape dimensions (768x512) work best for scenes"
            ]
        }


# Singleton instance
ghibli_service = GhibliImageService()
