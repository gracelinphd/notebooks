"""
Run basic inference, using vLLM to take advantage of PagedAttention.

Source: https://modal.com/docs/examples/vllm_mixtral

Run locally:
>modal run -q vllm_mixtral.py

To deploy:
>modal deploy vllm_mixtral.py

Example endpoint: https://modal.com/gracelinphd/main/apps/deployed/example-vllm-mixtral

Running Mixtral 8x7B Instruct model.
Expect ~3 minute cold starts. 
For a single request, the throughput is over 50 tokens/second. 
The larger the batch of prompts, the higher the throughput (up to hundreds of tokens per second).

"""

## Setup

import os
import time
import modal

MODEL_DIR = "/model"
MODEL_NAME = "mistralai/Mixtral-8x7B-Instruct-v0.1"
MODEL_REVISION = "1e637f2d7cb0a9d6fb1922f305cb784995190a83"
GPU_CONFIG = modal.gpu.A100(size="80GB", count=2)

## Define a container image
# Create a Modal image with model weights pre-saved to a directory.
# This is so the container doesn't need to re-download the model from HF.

## Download weights
# HF's function snapshot_download
# Mixtral is a "gated" model - a model with access requests enabled.
# Need to setup "HF_TOKEN" environment with HF acccess token - created a read access token.
# Also make sure to check agreement on HF: https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1.
# Mixtral is beefy, at nearly 100 GB in safetensors format, so this can take some time — at least a few minutes.
def download_model_to_image(model_dir, model_name, model_revision):
    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache

    os.makedirs(model_dir, exist_ok=True)

    snapshot_download(
        model_name,
        revision=model_revision,
        local_dir=model_dir,
        ignore_patterns=["*.pt", "*.bin"],  # Using safetensors
    )
    move_cache()


## Image definition
# Dockerhub image recommended by vLLM.
# Use "run_function" to run the function defined above, "download_model_to_image".
vllm_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "vllm==0.4.0.post1",
        "torch==2.1.2",
        "transformers==4.39.3",
        "ray==2.10.0",
        "hf-transfer==0.1.6",
        "huggingface_hub==0.22.2",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(
        download_model_to_image,
        timeout=60 * 20,
        kwargs={
            "model_dir": MODEL_DIR,
            "model_name": MODEL_NAME,
            "model_revision": MODEL_REVISION,
        },
        secrets=[modal.Secret.from_name("huggingface-secret")],
    )
)

app = modal.App(
    "example-vllm-mixtral"
)  # Note: prior to April 2024, "app" was called "stub"


## The model class
# Use @enter decorator.
# Load the model into memory just once every time a container starts up.
# Keep it cached on the GPU for each subsequent invocation of the function.
@app.cls(
    gpu=GPU_CONFIG,
    timeout=60 * 10,
    container_idle_timeout=60 * 10,
    allow_concurrent_inputs=10,
    image=vllm_image,
)
class Model:
    @modal.enter()
    def start_engine(self):
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine

        print("🥶 cold starting inference")
        start = time.monotonic_ns()

        engine_args = AsyncEngineArgs(
            model=MODEL_DIR,
            tensor_parallel_size=GPU_CONFIG.count,
            gpu_memory_utilization=0.90,
            enforce_eager=False,  # capture the graph for faster inference, but slower cold starts
            disable_log_stats=True,  # disable logging so we can stream tokens
            disable_log_requests=True,
        )
        self.template = "[INST] {user} [/INST]"

        # this can take some time!
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        duration_s = (time.monotonic_ns() - start) / 1e9
        print(f"🏎️ engine started in {duration_s:.0f}s")

    @modal.method()
    async def completion_stream(self, user_question):
        from vllm import SamplingParams
        from vllm.utils import random_uuid

        sampling_params = SamplingParams(
            temperature=0.75,
            max_tokens=128,
            repetition_penalty=1.1,
        )

        request_id = random_uuid()
        result_generator = self.engine.generate(
            self.template.format(user=user_question),
            sampling_params,
            request_id,
        )
        index, num_tokens = 0, 0
        start = time.monotonic_ns()
        async for output in result_generator:
            if (
                output.outputs[0].text
                and "\ufffd" == output.outputs[0].text[-1]
            ):
                continue
            text_delta = output.outputs[0].text[index:]
            index = len(output.outputs[0].text)
            num_tokens = len(output.outputs[0].token_ids)

            yield text_delta
        duration_s = (time.monotonic_ns() - start) / 1e9

        yield (
            f"\n\tGenerated {num_tokens} tokens from {MODEL_NAME} in {duration_s:.1f}s,"
            f" throughput = {num_tokens / duration_s:.0f} tokens/second on {GPU_CONFIG}.\n"
        )

    @modal.exit()
    def stop_engine(self):
        if GPU_CONFIG.count > 1:
            import ray

            ray.shutdown()


## Run the model locally
# Define a local_entrypoint to call our remote function sequentially for a list of inputs.
@app.local_entrypoint()
def main():
    questions = [
        "Implement a Python function to compute the Fibonacci numbers.",
        "What is the fable involving a fox and grapes?",
        "What were the major contributing factors to the fall of the Roman Empire?",
        "Describe the city of the future, considering advances in technology, environmental changes, and societal shifts.",
        "What is the product of 9 and 8?",
        "Who was Emperor Norton I, and what was his significance in San Francisco's history?",
    ]
    model = Model()
    for question in questions:
        print("Sending new request:", question, "\n\n")
        for text in model.completion_stream.remote_gen(question):
            print(text, end="", flush=text.endswith("\n"))

"""
Results from local run: 
Brackets with questions are added by me.

["Implement a Python function to compute the Fibonacci numbers."]

 Sure, here is a simple Python function to compute Fibonacci numbers using recursion:

```python
def fibonacci(n):
    if n <= 0:
        print("Input should be positive integer.")
    elif n == 1 or n == 2:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)
```

Please note that this function uses recursion and it's not the most efficient way to calculate Fibonacci numbers, especially for large `n`,
    Generated 128 tokens from mistralai/Mixtral-8x7B-Instruct-v0.1 in 3.5s, throughput = 37 tokens/second on GPU(A100-80GB, count=2).
Sending new request: What is the fable involving a fox and grapes?

["What is the fable involving a fox and grapes?"]

 The fable involving a fox and grapes is called "The Fox and the Grapes." It is one of Aesop's Fables, numbered 21 or 22 in the Perry Index. The story describes a fox who sees some high-hanging grapes and wishes to eat them. However, the grapes are too far out of reach, so the fox jumps as high as he can but still cannot get them. He then walks away, telling himself that the grapes were probably sour anyway.

The moral of this fable is: It is easy to desp
    Generated 128 tokens from mistralai/Mixtral-8x7B-Instruct-v0.1 in 1.9s, throughput = 69 tokens/second on GPU(A100-80GB, count=2).
Sending new request: What were the major contributing factors to the fall of the Roman Empire?

["What were the major contributing factors to the fall of the Roman Empire?"]

 The fall of the Roman Empire was the result of a complex mix of military, economic, and social factors, rather than any single cause. Some of the major contributing factors include:

1. Military challenges: External invasions by various Germanic and Asian tribes, such as the Visigoths, Ostrogoths, Huns, and Vandals, put significant pressure on the Roman military and stretched its resources thin. Internal issues, like political instability, corruption, and declining discipline within the army, further weakened Rome's ability to defend itself.

2. Economic problems: Overextension of
    Generated 128 tokens from mistralai/Mixtral-8x7B-Instruct-v0.1 in 1.9s, throughput = 69 tokens/second on GPU(A100-80GB, count=2).
Sending new request: Describe the city of the future, considering advances in technology, environmental changes, and societal shifts.

["Describe the city of the future, considering advances in technology, environmental changes, and societal shifts."]

 The city of the future is a harmonious blend of advanced technology, sustainable design, and progressive social structures, creating an inclusive, efficient, and eco-friendly urban environment. As vertical gardens and green roofs cover buildings, the air quality significantly improves, and urban agriculture becomes commonplace, providing fresh produce for residents.

Smart cities utilize artificial intelligence (AI), machine learning, and Internet of Things (IoT) technologies to optimize infrastructure, public services, and energy usage. Autonomous vehicles roam efficiently through well-planned streets, reducing traffic congestion and emissions. Intelligent grids
    Generated 128 tokens from mistralai/Mixtral-8x7B-Instruct-v0.1 in 3.2s, throughput = 40 tokens/second on GPU(A100-80GB, count=2).
Sending new request: What is the product of 9 and 8?

["What is the product of 9 and 8?"]

 The product of 9 and 8 is 72.

Here's how we can find the product:

1. Multiply 9 by the digit in the ones place of 8 (which is 8).
   So, 9 x 8 = 72.

Since there are no other digits in 8 to consider for further multiplication, our result is 72.
    Generated 91 tokens from mistralai/Mixtral-8x7B-Instruct-v0.1 in 1.3s, throughput = 68 tokens/second on GPU(A100-80GB, count=2).
Sending new request: Who was Emperor Norton I, and what was his significance in San Francisco's history?

["Who was Emperor Norton I, and what was his significance in San Francisco's history?"]

 Emperor Norton I, whose original name was Joshua Abraham Norton, was a famous and beloved figure in San Francisco's history during the mid-to-late 19th century. He was not an actual emperor but rather a self-proclaimed one who gained widespread recognition and acceptance from the city's residents.

Born in England in 1818, Norton moved to South Africa with his family when he was young. After working in various businesses there, he traveled to San Francisco in 1849 following the discovery of gold in California. Initially successful in real estate and rice spec
    Generated 128 tokens from mistralai/Mixtral-8x7B-Instruct-v0.1 in 1.9s, throughput = 68 tokens/second on GPU(A100-80GB, count=2).

"""

## Deploy and invoke the model
# To deploy:
# >modal deploy vllm_mixtral.py
#
# Invoke inference from other apps, sharing the same pool of GPU containers with all other apps we might need.
# Also see test_deployed_vllm_mixtral.py.
#
# $ python
# >>> import modal
# >>> f = modal.Function.lookup("example-vllm-mixtral", "Model.completion_stream")
# >>> for text in f.remote_gen("What is the story about the fox and grapes?"):
# >>>    print(text, end="", flush=text.endswith("\n"))
# 'The story about the fox and grapes ...
# ```


## Couple with a frontend web application
# We can stream inference from a FastAPI backend, also deployed on Modal.
#
# To serve:
# >modal serve vllm_mixtral.py

from pathlib import Path
from modal import Mount, asgi_app

#
# Frontend structure:
#
# ├── llm-frontend/
# │   ├── index.html
# │   ├── styles/
# │   │   └── main.css
# │   ├── scripts/
# │   │   └── app.js
#
frontend_path = Path(__file__).parent / "llm-frontend"

@app.function(
    mounts=[Mount.from_local_dir(frontend_path, remote_path="/assets")],
    keep_warm=1,
    allow_concurrent_inputs=20,
    timeout=60 * 10,
)
@asgi_app()
def vllm_mixtral():
    import json

    import fastapi
    import fastapi.staticfiles
    from fastapi.responses import StreamingResponse

    web_app = fastapi.FastAPI()

    @web_app.get("/stats")
    async def stats():
        stats = await Model().completion_stream.get_current_stats.aio()
        return {
            "backlog": stats.backlog,
            "num_total_runners": stats.num_total_runners,
            "model": MODEL_NAME + " (vLLM)",
        }

    @web_app.get("/completion/{question}")
    async def completion(question: str):
        from urllib.parse import unquote

        async def generate():
            async for text in Model().completion_stream.remote_gen.aio(
                unquote(question)
            ):
                yield f"data: {json.dumps(dict(text=text), ensure_ascii=False)}\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    web_app.mount(
        "/", fastapi.staticfiles.StaticFiles(directory="/assets", html=True)
    )
    return web_app

"""
After serving, you should see an app running on your modal dashboard.

In the message there should be something like:
├── Created vllm_mixtral => https://gracelinphd--example-vllm-mixtral-vllm-mixtral-dev.modal.run

Go to this website and chat with mistral.
The first question will have a cold-start, so it'll take a few min. 
But subsequent questions will run much faster.

TODO: even though I have my own llm-frontend/, the interface seems to be Modal's still. 
Not sure what's happening.
"""

