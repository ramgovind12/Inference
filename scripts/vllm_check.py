"""Send one request to a running vLLM server through VllmBackend.

Usage: uv run python scripts/vllm_check.py --url http://127.0.0.1:8100 --model <name>
Set INFERENCE_WORKER_VLLM_API_KEY if the server needs a key.
"""

import argparse
import asyncio
import os
import sys

from inference.backends.vllm import VllmBackend
from inference.core.errors import BackendError
from inference.core.schemas import InferenceRequest


async def run(url: str, model: str, prompt: str) -> int:
    backend = VllmBackend(url, api_key=os.environ.get("INFERENCE_WORKER_VLLM_API_KEY"))
    await backend.start()
    try:
        if not await backend.is_ready():
            print(f"vLLM is not ready: GET {url}/health failed", file=sys.stderr)
            return 1
        request = InferenceRequest(model=model, prompt=prompt, max_tokens=32)
        response = await backend.generate(request)
    except BackendError as exc:
        print(f"generation failed: {exc}", file=sys.stderr)
        return 1
    finally:
        await backend.stop()
    print(response.model_dump_json(indent=2))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="Say hello in one short sentence.")
    args = parser.parse_args()
    sys.exit(asyncio.run(run(args.url, args.model, args.prompt)))


if __name__ == "__main__":
    main()
