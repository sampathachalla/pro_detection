# Development tunnel

This Compose project runs the Vite development server and keeps a reverse SSH tunnel connected to `cloud-tunnel-gateway`.

The private key is mounted read-only at runtime. It is not copied into the image or this folder.

## Start

```sh
docker compose up -d --build
```

Local URL: `http://localhost:5173`

Mobile URL: `https://handwriting.34.57.225.191.nip.io`

## Status

```sh
docker compose ps
docker compose logs --tail=50 tunnel
```

The services use `restart: unless-stopped`, so Docker restarts them after a Docker Desktop restart. The mobile URL still depends on this Mac being online and connected to the internet.

## Stop

```sh
docker compose down
```
