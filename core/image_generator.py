import os
import time
import requests
import base64
import re
from pathlib import Path
from typing import Optional, Dict, Any, Union
from enum import Enum
import asyncio
import aiohttp
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

BFL_API_KEY = os.getenv("BFL_API_KEY")

class FluxModel(Enum):
    """Available FLUX models"""
    # Pro models
    FLUX_1_1_PRO = "flux-pro-1.1"
    FLUX_1_1_PRO_ULTRA = "flux-pro-1.1-ultra"

    # Core models
    FLUX_1_PRO = "flux-pro-1.0"
    FLUX_1_DEV = "flux-dev"
    FLUX_1_SCHNELL = "flux-schnell"
    
    # Kontext models (image editing)
    FLUX_1_KONTEXT_PRO = "flux-kontext-pro"
    FLUX_1_KONTEXT_MAX = "flux-kontext-max"
    FLUX_1_KONTEXT_DEV = "flux-kontext-dev"

    # Finetuned models
    FLUX_PRO_FINETUNED = "flux-pro-1.1-finetuned"
    FLUX_ULTRA_FINETUNED = "flux-pro-1.1-ultra-finetuned"


class FluxAPIAdapter:
    """
    Adapter class for Black Forest Labs FLUX API
    Handles image generation, editing, and local saving
    """
    
    def __init__(
        self, 
        use_raw_mode: bool,
        api_key: Optional[str] = None,
        base_url: str = "https://api.bfl.ai",
        local_save_dir: Optional[str] = "./generated_images"
    ):
        self.api_key = api_key or BFL_API_KEY
        if not self.api_key:
            raise ValueError("API key required. Set BFL_API_KEY env var or pass api_key")
        
        self.base_url = base_url
        self.local_save_dir = Path(local_save_dir) if local_save_dir else None
        
        if self.local_save_dir:
            self.local_save_dir.mkdir(parents=True, exist_ok=True)
        
        self.session = requests.Session()
        self.session.headers.update({
            'accept': 'application/json',
            'x-key': self.api_key,
            'Content-Type': 'application/json'
        })

        self.use_raw_mode = use_raw_mode

    def generate_image(
        self,
        prompt: str,
        model: FluxModel = FluxModel.FLUX_1_KONTEXT_MAX,
        width: int = 1024,
        height: int = 1024,
        aspect_ratio: Optional[str] = None,
        safety_tolerance: int = 6,
        seed: Optional[int] = None,
        save_locally: bool = True,
        prompt_upsampling: bool = False,
        filename: Optional[str] = None,
        poll_timeout: int = 120,  # bumped to 120s total poll wait
    ) -> Dict[str, Any]:
        """
        Generate an image using FLUX models

        Args:
            prompt: Text description of desired image
            model: FLUX model to use
            width: Image width (ignored if aspect_ratio is set)
            height: Image height (ignored if aspect_ratio is set)
            aspect_ratio: Image aspect ratio (e.g., "16:9", "1:1")
            safety_tolerance: Safety filtering level (1-6)
            seed: Random seed for reproducibility
            save_locally: Whether to save image locally
            filename: Custom filename for saved image
            poll_timeout: Maximum seconds to wait while polling for completion

        Returns:
            Dict containing image URL, local path (if saved), and metadata
        """
        print(f"\n\n==== GENERATING NEW IMAGE ====\n\n")

        payload = {
            "prompt": prompt,
            "safety_tolerance": safety_tolerance,
            "prompt_upsampling": prompt_upsampling,
            "raw": self.use_raw_mode
        }

        # Handle dimensions
        if aspect_ratio:
            payload["aspect_ratio"] = aspect_ratio
        else:
            payload["width"] = width
            payload["height"] = height

        if seed is not None:
            payload["seed"] = seed

        # Submit generation request with a network timeout
        endpoint = f"{self.base_url}/v1/{model.value}"
        last_exc = None
        for attempt in range(3):
            try:
                # (connect_timeout=5s, read_timeout=60s)
                response = self.session.post(endpoint, json=payload, timeout=(10, 120))
                response.raise_for_status()
                break
            except requests.exceptions.ReadTimeout as e:
                last_exc = e
                if attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise
            except requests.RequestException:
                raise
        data = response.json()

        request_id = data['id']
        polling_url = data.get('polling_url', f"{self.base_url}/v1/get_result")

        # Poll for completion with bounded wait
        result = self._poll_for_result(polling_url, request_id, max_wait=poll_timeout)

        # Save locally if requested
        local_path = None
        if save_locally and self.local_save_dir:
            local_path = self._save_image_locally(
                result['result']['sample'],
                filename or f"flux_{model.value}_{request_id}.png"
            )

        return {
            "image_url": result['result']['sample'],
            "local_path": str(local_path) if local_path else None,
            "request_id": request_id,
            "model": model.value,
            "prompt": prompt,
            "metadata": result.get('result', {})
        }



    def edit_image(
        self,
        image_path: str,
        prompt: str,
        model: FluxModel = FluxModel.FLUX_1_KONTEXT_MAX,
        guidance_scale: float = 2.5,
        prompt_upsampling: bool = False,
        save_locally: bool = True,
        filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Edit an existing image using FLUX Kontext models
        
        Args:
            image_path: Path to input image (local file or URL)
            prompt: Edit instruction
            model: FLUX Kontext model to use
            guidance_scale: How strongly to follow the prompt
            save_locally: Whether to save edited image locally
            filename: Custom filename for saved image
            
        Returns:
            Dict containing edited image URL, local path, and metadata
        """
        if model not in [FluxModel.FLUX_1_KONTEXT_PRO, FluxModel.FLUX_1_KONTEXT_MAX, FluxModel.FLUX_1_KONTEXT_DEV]:
            raise ValueError("Image editing requires a FLUX Kontext model")
        
        # Encode image to base64 if it's a local file
        if os.path.exists(image_path):
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            image_url = f"data:image/png;base64,{image_data}"
        else:
            image_url = image_path  # Assume it's already a URL
        
        payload = {
            "prompt": prompt,
            "input_image": image_url,  # Changed from "image" to "input_image"
            "guidance_scale": guidance_scale,
            "prompt_upsampling": prompt_upsampling
        }
        
        # Submit edit request - use flux-kontext-pro endpoint format
        if model == FluxModel.FLUX_1_KONTEXT_PRO:
            endpoint = f"{self.base_url}/v1/flux-kontext-pro"
        elif model == FluxModel.FLUX_1_KONTEXT_MAX:
            endpoint = f"{self.base_url}/v1/flux-kontext-max"
        else:  # FLUX_1_KONTEXT_DEV
            endpoint = f"{self.base_url}/v1/flux-kontext-dev"
            
        response = self.session.post(endpoint, json=payload)
        response.raise_for_status()
        
        data = response.json()
        request_id = data['id']
        polling_url = data.get('polling_url', f"{self.base_url}/v1/get_result")
        
        # Poll for completion
        result = self._poll_for_result(polling_url, request_id)
        
        # Save locally if requested
        local_path = None
        if save_locally and self.local_save_dir:
            local_path = self._save_image_locally(
                result['result']['sample'],
                filename or f"flux_edit_{model.value}_{request_id}.png"
            )
        
        return {
            "image_url": result['result']['sample'],
            "local_path": str(local_path) if local_path else None,
            "request_id": request_id,
            "model": model.value,
            "prompt": prompt,
            "input_image": image_path,
            "metadata": result.get('result', {})
        }

    def _poll_for_result(self, polling_url: str, request_id: str, max_wait: int = 900) -> Dict[str, Any]:
        """Poll for generation result"""
        start_time = time.time()

        while time.time() - start_time < max_wait:
            time.sleep(0.5)

            try:
                response = self.session.get(
                    polling_url,
                    params={'id': request_id},
                    timeout=5  # avoid hanging on each poll
                )
                response.raise_for_status()
                result = response.json()
            except requests.exceptions.Timeout:
                continue
            except requests.RequestException as e:
                print(f"Warning polling result: {e}, retrying...")
                continue

            if result.get('status') == 'Ready':
                return result
            elif result.get('status') in ['Error', 'Failed']:
                raise Exception(f"Generation failed: {result}")

        raise TimeoutError(f"Request {request_id} timed out after {max_wait} seconds")



    def _save_image_locally(self, image_url: str, filename: str) -> Path:
        """Download and save image locally"""
        if self.local_save_dir is None:
            raise ValueError("local_save_dir is None - cannot save locally")
        
        file_path = self.local_save_dir / filename

        if image_url.startswith("data:"):
            # decode base64 data URL
            try:
                header, b64 = image_url.split(",", 1)
                data = base64.b64decode(b64)
                with open(file_path, 'wb') as f:
                    f.write(data)
                return file_path
            except Exception as e:
                raise RuntimeError(f"Failed to decode data URL image: {e}")
            
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        
        with open(file_path, 'wb') as f:
            f.write(response.content)
        
        return file_path




# Usage example
if __name__ == "__main__":
    # Initialize adapter
    flux = FluxAPIAdapter(use_raw_mode=False, local_save_dir="./my_images")
    
    # Generate an image
    result = flux.generate_image(
        prompt="A futuristic cityscape at sunset",
        model=FluxModel.FLUX_1_1_PRO,
        aspect_ratio="16:9"
    )
    print(f"Generated image: {result['local_path']}")
    
    # Edit the generated image
    if result['local_path']:
        edit_result = flux.edit_image(
            image_path=result['local_path'],
            prompt="Add flying cars to the scene",
            model=FluxModel.FLUX_1_KONTEXT_PRO
        )
        print(f"Edited image: {edit_result['local_path']}")
