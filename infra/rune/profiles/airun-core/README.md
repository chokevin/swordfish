# airun core profiles — vendored snapshot

These three files (`sandbox.yaml`, `serve.yaml`, `train.yaml`) are vendored
from the airun-zero exploration at the user's V0 kit:

> `~/.copilot/session-state/.../files/airun-zero/profiles/core/`

They are the canonical airun core catalog — tenant-agnostic profiles that
all packs (including `swordfish-pack.yaml` in the parent directory)
extend via `spec.extends:`.

**Why vendor?** Rune resolves `extends:` client-side by looking up the
parent profile in the configured search path. Without these files
co-located, `rune profile show swordfish-bench-a100` fails with
`extends unknown profile "ai-train-gpu-l"`. Vendoring keeps the swordfish
checkout self-contained.

**Update policy:** if airun-zero ships changes to the core profiles,
re-vendor by re-copying. Track upstream divergence in PR descriptions
(no automatic sync today).

**Provenance:** the airun-zero V0 kit lives in the AGENT.md-managed
session workspace; it is the user's own work product exploring the
airun PRD applied to voice-agent-flex. Not external IP.
