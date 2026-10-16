#!/usr/bin/env bash
# Existing-account job management only; scientific work stays in course jobs.
set -euo pipefail
TASK_CONFIG=/private/tmp/phase1-hpc-control/config
TASK_KNOWN=/private/tmp/phase1-hpc-control/known_hosts
TASK_SOCKET=/private/tmp/phase1-hpc-control/cloud-admin.sock
TASK_EXPECTED='10.144.0.50 ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIAYH/TLrQjhQr1HvwFn8FydP3AimeQim61Z+Z0xJGGjr'
[[ -r "$TASK_CONFIG" && -r "$TASK_KNOWN" ]]
TASK_ANCHOR=$(ssh-keygen -F 10.144.0.50 -f "$TASK_KNOWN" | sed '/^#/d')
[[ "$TASK_ANCHOR" == "$TASK_EXPECTED" ]] || {
  echo 'The verified public host anchor changed. Stop and ask the agent to inspect.'
  exit 2
}
TASK_SSH=(ssh -F "$TASK_CONFIG" -o StrictHostKeyChecking=yes -o ForwardAgent=no
  -o ControlPath="$TASK_SOCKET")
if "${TASK_SSH[@]}" -O check phase1-cloud-admin 2>/dev/null; then
  exec "${TASK_SSH[@]}" -o BatchMode=yes phase1-cloud-admin hostname
fi
echo 'Verified normal portal host: ood-burst-1, SHA256:LG/AJU6ZmBI3WIFpcy+pquCuquGOoMQv/AaOd0exMR4'
echo 'Complete the normal NYU device sign-in locally; then press Enter in this terminal.'
echo 'No password, PIN, or verification code should be sent to chat.'
exec "${TASK_SSH[@]}" -o PubkeyAuthentication=no \
  -o PreferredAuthentications=keyboard-interactive -o NumberOfPasswordPrompts=1 \
  -o ConnectTimeout=10 -o ConnectionAttempts=1 \
  -o ControlMaster=yes -o ControlPersist=8h -o ServerAliveInterval=60 \
  phase1-cloud-admin hostname
