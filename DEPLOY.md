# Running the whole stack (Docker + GPU + Cloudflare tunnel)

One command brings up the detector (on the GPU), the web client, a reverse proxy,
and a Cloudflare tunnel that exposes it all over HTTPS so phones can use their
camera and GPS (browsers only allow those on a secure origin).

```bash
docker compose up --build
```

Services:

| service | what it is | exposed |
|---|---|---|
| `detector` | FastAPI + YOLO + fusion, **runs on the GPU** | internal `:8000` |
| `web` | Next.js operator/device client | internal `:3000` |
| `proxy` | Caddy — one origin in front of both | host `:8080` (LAN) |
| `cloudflared` | Cloudflare tunnel → HTTPS | public URL |

## Prerequisites (GPU)

The detector needs the host's NVIDIA GPU:

- **NVIDIA driver** on the host.
- **NVIDIA Container Toolkit** so Docker can pass the GPU in
  (`nvidia-ctk`); test with:
  ```bash
  docker run --rm --gpus all nvidia/cuda:12.6.2-base-ubuntu22.04 nvidia-smi
  ```
- **Windows:** Docker Desktop with the WSL2 backend + a WSL2-capable NVIDIA
  driver. The same `nvidia-smi` test above should print your GPU.

No GPU handy? Set `BATTLESIGHT_DEVICE: "cpu"` on the `detector` service in
`docker-compose.yml` and drop its `deploy.resources` block — it will run on CPU
(slower, but everything works).

## Get the public URL

The tunnel prints a `https://<random>.trycloudflare.com` URL on startup:

```bash
docker compose logs -f cloudflared
```

Then, on the phones (any network — it's public HTTPS):

- Operator overview: `https://<random>.trycloudflare.com/operator`
- Device / AR client: `https://<random>.trycloudflare.com/device/alpha`
  (use a different id per phone: `/device/alpha`, `/device/bravo`, …)

Tap **Start**, allow camera + location + motion, and stand two phones close
together pointing at the same scene. Each phone's screen shows only the **ghost
boxes** backfilled from other feeds; the operator page shows the full video wall
(own boxes + ghosts) and the unified world map.

On the laptop itself you can also use the LAN port without the tunnel — e.g.
`http://<laptop-ip>:8080/operator` — but phone camera/GPS still need the HTTPS
tunnel URL.

## Verify the GPU is actually being used

```bash
docker compose exec detector python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
curl -s http://localhost:8080/health   # "device" should be "0" / cuda, not "cpu"
```

## Optional: a stable named tunnel instead of the random URL

The quick tunnel's URL changes every restart. For a fixed hostname on your own
Cloudflare-managed domain, create a tunnel in the Zero Trust dashboard, copy its
token, and replace the `cloudflared` command in `docker-compose.yml`:

```yaml
  cloudflared:
    image: cloudflare/cloudflared:latest
    depends_on: [proxy]
    command: tunnel --no-autoupdate run --token ${TUNNEL_TOKEN}
    restart: unless-stopped
```

Put `TUNNEL_TOKEN=...` in a `.env` file next to `docker-compose.yml`, and in the
dashboard point the tunnel's public hostname at `http://proxy:80`.

## Notes

- Build context for the detector image is the **repo root** (not `detector/`),
  because `detector/app/__init__.py` puts the sibling `fusion/` package on the
  path — both dirs must share a root in the image, mirroring the source tree.
- CUDA 12.6 base matches the pinned `torch==2.13.0+cu126` in
  `detector/requirements.txt`.
