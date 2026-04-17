"""
This is the single file that knows how to talk to an LLM. Locally it talks to Ollama,
in prod it talks to Google AI. Every other file just calls this, which makes it eaiser
to swap from local to prod.

LLM clien - abstracts over Ollama (local) and Google AI (prod).

Locally: uses Ollama with gemma3:4b.
Prod: uses Google AI API with gemma-2.0-flash.

To swithc environments, change LLM_PROVIDER in .env:
    LLM_PROVIDER=ollama  # for local development
    LLM_PROVIDER=google  # for productions
"""

import logging
from typing import Optional
import ollama
import httpx

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class LLMClient:
    """
    Unified interface for LLM inference.
    AUtomaitcally routes to Ollama or Google AI based on settings.
    """

    async def complete(self, prompt: str, system: Optional[str] = None, image_bytes: Optional[bytes] = None,) -> str:
        """
        Send a prompt to the LLM and get a response.

        Args:
            prompt: The user message.
            system: Optional system prompt.
            image_bytes: Optional image to send (multimodal).
        
        Returns:
            The LLM's response as a string.
        """
        if settings.llm_provider == "google":
            return await self._complete_google(prompt, system, image_bytes)
        else:
            return await self._complete_ollama(prompt, system, image_bytes)
        
    # -------------------------------------------------------------------------------------------------------------------
    # Ollama (local)
    # -------------------------------------------------------------------------------------------------------------------

    async def _complete_ollama(self, prompt: str, system: Optional[str], image_bytes: Optional[bytes]) -> str:
        """
        Call local Ollama server with gemma2:4b.
        Runs in a thread pool since Ollama client is asynchronous.
        """
        import asyncio

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._ollama_sync(prompt, system, image_bytes)
        )
        return response
    
    def _ollama_sync(self, prompt: str, system: Optional[str], image_bytes: Optional[bytes]) -> str:
        """
        Synchronous call to Ollama client. Used internally by the async version.
        """
        messages = []

        if system:
            messages.append({"role": "system", "content": system})
        
        user_message: dict = {"role": "user", "content": prompt}

        # Attach image if provided - Gemma 3 is multimodal, so it can accept images as input.
        if image_bytes:
            import base64
            from PIL import Image
            import io

            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            # Resize to max 512px on longest side — sufficient for detection
            # full resolution images make inference very slow on CPU
            max_size = 512
            ratio = min(max_size / img.width, max_size / img.height)
            if ratio < 1:
                new_size = (int(img.width * ratio), int(img.height * ratio))
                img = img.resize(new_size, Image.LANCZOS)
            
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=75)
            jpeg_bytes = buffer.getvalue()
            user_message["images"] = [base64.b64encode(jpeg_bytes).decode()]
        
        messages.append(user_message)

        logger.info("Calling Ollama model=%s has_image=%s", settings.llm_model_local, image_bytes is not None)

        response = ollama.chat(
            model=settings.llm_model_local,
            messages=messages,
            options={"num_predict": 512}, # Limit output tokens.
        )
        return response["message"]["content"]
    
    # -------------------------------------------------------------------------------------------------------------------
    # Google AI (prod)
    # -------------------------------------------------------------------------------------------------------------------

    async def _complete_google(self, prompt: str, system: Optional[str], image_bytes: Optional[bytes]) -> str:
        """
        Call Google AI API with gemma-2.0-flash.
        Uses httpx for async HTTP.
        """
        import json

        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": settings.google_api_key,
        }

        # Build content parts.
        parts = []
        if image_bytes:
            import base64
            parts.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": base64.b64encode(image_bytes).decode(),
                }
            })
        parts.append({"text": prompt})

        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "temperature": 0.1,     # low temmp = more consistent reasoning.
                "maxOutputTokens": 1024,
            }
        }

        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{settings.llm_model_google}:generateContent"

        logger.info("Calling Google AI model=%s has_image=%s", settings.llm_model_google, image_bytes is not None)

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
        
        return data["candidates"][0]["content"]["parts"][0]["text"]