"""Run a read-only, locally mounted Kev checkpoint and base model in a container."""

import os
from pathlib import Path

import uvicorn

from kev import serve
from kev.checkpoint import Checkpoint, LoadOptions
from kev.device import default_device


def main():
    model_root = Path(os.environ["KEV_MODEL_ROOT"])
    checkpoint = Checkpoint(model_root / "kev-4b")
    checkpoint.meta.base = str(model_root / "Qwen3.5-4B-Base")
    checkpoint.meta.base_revision = None

    device = default_device()
    if device != "cuda":
        raise RuntimeError("CUDA GPU is not available inside the container")

    tokenizer, model = checkpoint.load(device, LoadOptions.from_env())
    serve.app.state.server = serve.Server(checkpoint, tokenizer, model, device)
    uvicorn.run(serve.app, host="0.0.0.0", port=8009)


if __name__ == "__main__":
    main()
