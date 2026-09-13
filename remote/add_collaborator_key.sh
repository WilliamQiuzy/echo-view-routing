#!/usr/bin/env bash
# Grant a collaborator SSH access by appending THEIR public key to the shared account's authorized_keys.
#   remote/add_collaborator_key.sh "ssh-ed25519 AAAA... name@laptop"
# Never share the private key; each person generates their own pair (ssh-keygen -t ed25519) and sends the .pub line.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
[[ $# -eq 1 ]] || { echo "usage: $0 '<public key line>'" >&2; exit 2; }
KEY="$1"
[[ "$KEY" =~ ^(ssh-ed25519|ssh-rsa|ecdsa-sha2-nistp256)\ [A-Za-z0-9+/=]+ ]] || { echo "error: not a public key line" >&2; exit 1; }
rssh "mkdir -p ~/.ssh && chmod 700 ~/.ssh && touch ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys && \
  if grep -qF '$(echo "$KEY" | awk '{print $2}')' ~/.ssh/authorized_keys; then echo 'already present'; else echo '$KEY' >> ~/.ssh/authorized_keys && echo 'added'; fi; \
  echo \"authorized keys now: \$(wc -l < ~/.ssh/authorized_keys)\""
