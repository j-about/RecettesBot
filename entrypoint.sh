#!/bin/sh
set -e

# Unset empty ANTHROPIC_API_KEY so the Claude SDK falls back to credentials file.
if [ -z "$ANTHROPIC_API_KEY" ]; then
    unset ANTHROPIC_API_KEY
fi

# Copy Claude credentials if mounted (bind mount may have wrong ownership).
CRED_SRC="/mnt/claude-credentials/.credentials.json"
CRED_DST="/home/app/.claude/.credentials.json"
if [ -f "$CRED_SRC" ]; then
    mkdir -p /home/app/.claude
    cp "$CRED_SRC" "$CRED_DST"
    chown app:app /home/app/.claude "$CRED_DST"
    chmod 600 "$CRED_DST"
    echo "Claude credentials copied from mount."
fi

exec gosu app "$@"
