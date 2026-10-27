# Kev-4B Docker deployment

Build on a networked Linux `amd64` machine with Docker. The models stay outside
the images. The API image uses the versions exported from `uv.lock`; regenerate
the requirements file after changing the lock:

```bash
uv export --frozen --no-dev --extra serve --no-hashes --no-emit-project \
  -o docker/requirements.serve.txt
docker build -f docker/Dockerfile.api -t kev-api:local .
docker build -t kev-playground:local ./playground
mkdir -p dist
docker save -o dist/kev-images.tar kev-api:local kev-playground:local
```

Copy `dist/kev-images.tar` and `compose.yaml` to a separate deployment directory
on the server, for example `/data1/guoyan/kev-docker/`. This leaves the existing
project checkout alone. Put the two models under the same absolute directory,
for example:

```text
/data1/guoyan/kev-main/models/
├── kev-4b/                 # head.pt, adapter_config.json, adapter_model.safetensors
└── Qwen3.5-4B-Base/        # config.json, tokenizer and every weight shard
```

The base model must be the revision recorded in the Kev-4B checkpoint:
`1001bb4d826a52d1f399e183466143f4da7b741b`. The container mounts the
whole model root at the **same absolute path** as the host. This also preserves
absolute symlinks into a cache under that root. Any links pointing outside the
root must be resolved or their targets mounted too.

On the server, from the directory containing `compose.yaml`:

```bash
docker load -i kev-images.tar
cat > .env <<'EOF'
MODEL_ROOT=/data1/guoyan/kev-main/models
GPU_ID=4
EOF
docker compose config --quiet
docker compose up -d --no-build
docker compose ps
curl -fsS http://127.0.0.1:3001/kev/v1/models
```

Set `GPU_ID` to an available host GPU index. Open
`http://SERVER_LAN_IP:3001` from an internal browser. If startup fails, inspect
`docker compose logs --tail=100 kev-api` and
`docker compose logs --tail=100 kev-playground`. The API is reachable only on
the Compose network; the published port is the web page on `3001`.
