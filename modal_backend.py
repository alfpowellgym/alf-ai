import io
import random
from typing import Optional

import modal

app = modal.App("alf-ai-image-backend")

MINUTES = 60
CACHE_DIR = "/cache"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "accelerate==0.33.0",
        "diffusers==0.31.0",
        "fastapi[standard]==0.115.4",
        "huggingface-hub==0.36.0",
        "sentencepiece==0.2.0",
        "torch==2.5.1",
        "torchvision==0.20.1",
        "transformers~=4.44.0",
    )
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_HUB_CACHE": CACHE_DIR,
        }
    )
)

with image.imports():
    import diffusers
    import torch
    from fastapi import Response


MODEL_ID = "adamo1139/stable-diffusion-3.5-large-turbo-ungated"
MODEL_REVISION_ID = "9ad870ac0b0e5e48ced156bb02f85d324b7275d2"

cache_volume = modal.Volume.from_name(
    "alf-ai-model-cache",
    create_if_missing=True,
)


@app.cls(
    image=image,
    gpu="H100",
    timeout=10 * MINUTES,
    volumes={CACHE_DIR: cache_volume},
)
class ImageGenerator:

    @modal.enter()
    def load_model(self):
        print("Loading ALF AI image model...")

        self.pipe = (
            diffusers.StableDiffusion3Pipeline.from_pretrained(
                MODEL_ID,
                revision=MODEL_REVISION_ID,
                torch_dtype=torch.bfloat16,
            )
            .to("cuda")
        )

        print("ALF AI image model ready.")


    @modal.method()
    def generate(
        self,
        prompt: str,
        seed: Optional[int] = None,
    ) -> bytes:

        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        torch.manual_seed(seed)

        realism_prompt = (
            "ultra photorealistic professional photograph, "
            "true-to-life lighting, highly realistic skin and textures, "
            "natural anatomy, physically realistic materials, "
            "subtle imperfections, believable depth and detail, "
            "not CGI, not illustration, not digital art. "
        )

        final_prompt = realism_prompt + prompt

        result = self.pipe(
            final_prompt,
            num_images_per_prompt=1,
            num_inference_steps=4,
            guidance_scale=0.0,
            max_sequence_length=512,
        ).images[0]

        with io.BytesIO() as buffer:
            result.save(buffer, format="PNG")
            output = buffer.getvalue()

        torch.cuda.empty_cache()

        return output


    @modal.fastapi_endpoint(docs=True)
    def web(
        self,
        prompt: str,
        seed: Optional[int] = None,
    ):
        image_bytes = self.generate.local(
            prompt=prompt,
            seed=seed,
        )

        return Response(
            content=image_bytes,
            media_type="image/png",
            headers={
                "Access-Control-Allow-Origin": "*",
                "Cache-Control": "no-store",
            },
        )
