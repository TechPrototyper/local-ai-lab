# The full triad ran on the 5090 — and why production rolled it back (2026-08-24)

**The triad:** GridBook 13 GB weights (FP8-CB product codebook) + NVFP4 KV
cache + DFlash2 speculative decoding — the exact combination this lab's
memory-bound track has been driving toward: smallest quality-holding
weights, 4-bit KV, and block-diffusion speculation, together on a 32 GB
consumer card.

## What happened (ops log, 2026-08-24)

- **~09:08** — a `vllm-gb-nvfp4` pod went live on the RTX 5090:
  GridBook 13 GB target, `--kv-cache-dtype nvfp4`, an **877k-token KV
  context configuration** (the ~11 GB freed by the smaller weights went
  straight into the KV pool — roughly double the AQUA-era 469k headroom),
  DFlash2 speculation on top. Exposed through the gateway as model `gb`.
- **~09:58** — briefly promoted to the **primary route** for
  `qwen3.8-27b` traffic.
- **~10:17** — rolled back ("Entscheidung B"): RTX returned to AQUA,
  no-spec, NVFP4-KV, prefix caching on.

## Why the rollback — and why it is *not* about the triad

The rollback reason was a single, precise incompatibility, live-confirmed:
**prefix caching gets 0% hits under DFlash2 speculation on hybrid-GDN
models** (identical prompt, zero hit). Upstream tracks this as
[vllm#52244](https://github.com/vllm-project/vllm/pull/52244)
("Restore hybrid GDN prefix-cache hits under MTP spec decoding") — open
and active as of 2026-08-26.

On the RTX — the lab's interactive **agent** tier, where harnesses re-send
huge contexts every turn — prefix caching is worth more than speculation:
losing the cache means paying full prefill per turn, which dwarfs a
+decode-speed win. So production chose cache over speculation. That is a
scheduling/caching gap in the current spec-decode path, not a limitation
of the weights/KV/speculation composition itself — which booted and
served.

## Where this leaves the triad

- **Composes technically:** all three components ran together on sm120
  (and since 2026-08-27 the spec×NVFP4-KV pair is validated paired and
  cross-arch — see
  [`dflash2-nvfp4-sm120-spec-serves.md`](dflash2-nvfp4-sm120-spec-serves.md)
  and the upstream package #53977/#53978/#53979).
- **Production adoption on the agent tier is gated on #52244** (or on
  workloads that don't lean on prefix caching — batch/single-shot jobs
  already fit).
- **Still unmeasured, honestly:** DFlash2 acceptance against the
  *GridBook* target (the drafter was trained against the BF16/AQUA
  distribution; greedy quality is unaffected by construction, but
  acceptance = speed could shift), a needle/battery pass at the far end of
  the 877k window, and GridBook's verdict-tier quality run (n=1319; the
  n=250 triage is in
  [`gridbook-13gb-quality-holds.md`](gridbook-13gb-quality-holds.md)).
