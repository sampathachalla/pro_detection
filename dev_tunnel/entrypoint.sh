#!/bin/sh
set -eu

: "${TUNNEL_HOST:?TUNNEL_HOST is required}"
: "${TUNNEL_USER:?TUNNEL_USER is required}"
: "${REMOTE_PORT:?REMOTE_PORT is required}"
: "${LOCAL_TARGET:?LOCAL_TARGET is required}"

exec ssh \
  -N \
  -T \
  -o BatchMode=yes \
  -o ConnectTimeout=10 \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -o StrictHostKeyChecking=accept-new \
  -i /run/secrets/tunnel_key \
  -R "127.0.0.1:${REMOTE_PORT}:${LOCAL_TARGET}" \
  "${TUNNEL_USER}@${TUNNEL_HOST}"
