# Deploying FusionSight

Two ways to run it, sharing the same images:

- **All-in-one** (one laptop) — everything on the compose network. Best for dev.
- **3-laptop split** — detector (GPU) / fusion / broadcast on separate machines,
  linked by an encrypted mesh. This is the demo topology.

## Services

| service | role | GPU | port |
|---|---|---|---|
| detector | YOLO inference + operator video relay; streams raw detections to fusion | yes | 8000 |
| fusion | pose + cross-feed ghosts + world map (all shared state) | no | 8100 |
| web | Next.js operator/device client | no | 3000 |
| proxy (Caddy) | one origin, routes paths to detector/fusion/web | no | 80 |
| cloudflared | Cloudflare tunnel → public HTTPS | no | — |

The phone/operator only ever see the proxy's one HTTPS origin. Only the light
detection JSON crosses detector→fusion; video frames stay on the detector.

---

## A) All-in-one (single laptop)

```bash
docker compose up --build
```

Public URL is printed by the tunnel:

```bash
docker compose logs -f cloudflared      # https://<random>.trycloudflare.com
```

Open `…/operator` on the laptop and `…/device/alpha`, `…/device/bravo`, … on the
phones. Requires the NVIDIA Container Toolkit (see GPU note below); set
`BATTLESIGHT_DEVICE=cpu` on the detector to run without a GPU.

---

## B) 3-laptop split

Each laptop runs **one** self-contained stack — `cd` into its folder and
`docker compose up --build`, nothing else to build.

```
deploy/detector/    -> Laptop 1 (GPU)
deploy/fusion/      -> Laptop 2
deploy/broadcast/   -> Laptop 3 (public edge)
```

### 1. One-time: encrypted mesh (Tailscale)

Install Tailscale on all three laptops and log into the same tailnet. Name the
machines **detector**, **fusion**, **broadcast** (Tailscale admin console →
rename) so the default hostnames in the compose files resolve — then you don't
have to edit any `.env`. All laptop↔laptop traffic is then WireGuard-encrypted,
even on the same LAN, with no open ports on the raw network.

> If you'd rather not use MagicDNS names, copy each folder's `.env.example` to
> `.env` and put the tailnet IPs in instead.

### 2. One-time: auth on the public edge (Cloudflare Access)

So only your people can reach the broadcast laptop: in the Cloudflare Zero Trust
dashboard, put an **Access** application (email OTP / SSO) in front of the
tunnel hostname. (Quick tunnels get a random hostname; use a **named tunnel**
with your own domain to attach Access — see the cloudflared note below.)

### 3. Start each laptop

```bash
# Laptop 1 (GPU)
cd deploy/detector && docker compose up --build

# Laptop 2
cd deploy/fusion && docker compose up --build

# Laptop 3 (edge)
cd deploy/broadcast && docker compose up --build
```

Get the public URL from the broadcast laptop:

```bash
cd deploy/broadcast && docker compose logs -f cloudflared
```

Phones open `…/device/<name>`, the operator opens `…/operator`.

### Data flow (why the split is safe for latency)

The phone screen shows **only ghosts**, which come from fusion on their own
channel — so the detector→fusion hop is *not* in the phone's own-detection path.
On one LAN/tailnet that hop is ~1–5 ms, so ghosts feel live.

```
phone frames ─▶ detector (infer) ─▶ fusion (correlate) ─▶ phone ghosts + operator + map
phone pose ───────────────────────▶ fusion
operator video ◀── detector (frames live here)
```

---

## GPU prerequisites (detector laptop)

- NVIDIA driver + **NVIDIA Container Toolkit** on the host. Verify:
  ```bash
  docker run --rm --gpus all nvidia/cuda:12.6.2-base-ubuntu22.04 nvidia-smi
  ```
- Windows: Docker Desktop with the WSL2 backend + a WSL2 NVIDIA driver.

Verify the detector is really on the GPU:
```bash
cd deploy/detector && docker compose exec detector python -c "import torch; print(torch.cuda.is_available())"
```

## Named Cloudflare tunnel (stable URL + Access)

The quick tunnel's URL changes each restart and can't take Cloudflare Access.
For a fixed hostname, create a tunnel in the dashboard, point its public
hostname at `http://proxy:80`, and swap the broadcast `cloudflared` command:

```yaml
    command: tunnel --no-autoupdate run --token ${TUNNEL_TOKEN}
```

with `TUNNEL_TOKEN=...` in `deploy/broadcast/.env`.

## Config reference

| var | service | meaning |
|---|---|---|
| `FUSION_WS_URL` | detector | fusion ingest URL (default `ws://fusion:8100/ws/ingest`) |
| `BATTLESIGHT_DEVICE` | detector | `0` = GPU, `cpu` = CPU |
| `FUSION_BASE_RADIUS_M` / `FUSION_WINDOW_MS` / `FUSION_POSE_MAX_AGE_MS` | fusion | cross-feed tuning |
| `DETECTOR_HOST` / `FUSION_HOST` | broadcast | mesh addresses of the other two laptops |
