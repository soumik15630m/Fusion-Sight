# LAN / HTTPS single-laptop demo

Runs the whole system on **one laptop**, served over **HTTPS on your local
network** — no internet, no Cloudflare tunnel. Fully isolated from the tunnel
demo (own project name `fusionlan`, own images, only port `8443` published), so
it won't disturb anything already running.

Phone camera + GPS require a *trusted* HTTPS origin, so there's a **one-time
per-phone step**: install Caddy's root certificate. After that it just works.

## 1. Find your laptop's LAN IP

```powershell
ipconfig
```
Take the IPv4 of your active Wi-Fi/Ethernet adapter (e.g. `192.168.1.50`).

## 2. Set it

```powershell
cd deploy\lan
copy .env.example .env
```
Edit `.env` → `LAN_HOST=192.168.1.50` (your IP).

## 3. Open the firewall (Administrator PowerShell)

```powershell
New-NetFirewallRule -DisplayName "FusionSight LAN" -Direction Inbound -Protocol TCP -LocalPort 8443 -Action Allow
```

## 4. Start it

```powershell
docker compose up --build -d
```
(First build reuses cached image layers, so it's quick.)

## 5. Export Caddy's root certificate

```powershell
docker compose exec proxy cat /data/caddy/pki/authorities/local/root.crt > caddy-root.crt
```
This `caddy-root.crt` is what each phone must trust.

## 6. Install the cert on each phone (one time)

Send `caddy-root.crt` to the phone (email / USB / any file transfer), then:

- **Android:** Settings → Security → *Encryption & credentials* → *Install a
  certificate* → *CA certificate* → pick the file.
- **iPhone:** open the file → *Install Profile* → then Settings → General →
  About → *Certificate Trust Settings* → enable full trust for it.

## 7. Open the app

- **Phones:** `https://<LAN_HOST>:8443/device/alpha` (…/bravo, /charlie, /delta,
  /echo). Set a name → **Start** → allow camera + location + motion.
- **Operator:** `https://<LAN_HOST>:8443/operator`

Verify from the laptop first:
```powershell
curl.exe -sk https://<LAN_HOST>:8443/health
```
(`-k` skips the cert check for this one test.) You want the detector JSON with
`"fusion_link":{...,"connected":true}`.

## Notes

- **GPU memory:** this runs its own detector model. On a 6 GB GPU, running this
  *at the same time* as the other backend stack may run out of memory — stop the
  other stacks first (`docker compose down` in their folders) if inference fails
  to load.
- No URL churn here — the address is your fixed LAN IP.
- Everything stays on the LAN; nothing is exposed to the internet.
