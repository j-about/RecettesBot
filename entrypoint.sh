#!/bin/sh
set -e

# Unset empty auth env vars so the Claude CLI falls through the auth chain:
#   ANTHROPIC_API_KEY -> CLAUDE_CODE_OAUTH_TOKEN -> ~/.claude/.credentials.json
if [ -z "$ANTHROPIC_API_KEY" ]; then
    unset ANTHROPIC_API_KEY
fi
if [ -z "$CLAUDE_CODE_OAUTH_TOKEN" ]; then
    unset CLAUDE_CODE_OAUTH_TOKEN
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
