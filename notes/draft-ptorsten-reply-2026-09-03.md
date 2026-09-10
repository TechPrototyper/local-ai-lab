# ENTWURF — Antwort an ptorsten (Meta) auf #46329 — POSTEN NUR MIT TIMS GO

Kontext: ptorsten bestätigte 09-03 ~22:37 UTC unabhängig die #53979-
Problemklasse (per-KV-Group-Scoping), erklärte unseren Sweep-Einbruch
(split-KV off + eager verify) und bot Repro-Hilfe an. Spock-Ton: Fakten,
Querverweise, konkretes Angebot, keine Bewertung.

Offene Entscheidungen für Tim (vor dem Post ankreuzen):
- [ ] Repro-Angebot annehmen und unseren Harness teilen? (scripts/kvsweep/
      decode_context_sweep.py ist klein und teilbar)
- [ ] GB10-Crash-Datenpunkt (CUBLAS 09-02) beisteuern? (stützt sein #46307)
- [ ] #53979 aktiv verlinken (= unser PR; er hat das Problem unabhängig
      gescoped — Fold-in-Frage an ihn oder jethac überlassen?)

---

> Thanks for the depth here — three cross-checks from our side:
>
> 1. Builder scope: your per-KV-group fix matches what we carry as
>    #53979 (non-causal FA2 prefill wrapper for the DFlash family +
>    SWA guard, rebased onto `9590d2464`, head `71a1133e8`). Two
>    independent implementations of the same scoping conclusion.
>
> 2. Your split-KV explanation fits our table above: 14× decode decline
>    1k→262k with draft acceptance flat at 0.58–0.67 (mtp@2, single
>    stream). Acceptance rules out drafter quality; a verify pass that
>    streams the full KV from one CTA row predicts exactly this shape.
>    We can re-run our sweep with any patched FlashInfer build on
>    RTX 5090 (sm120) and GB10 (sm121) within a day — the harness is
>    ~150 lines, happy to share.
>
> 3. GB10 memory: one supporting datapoint for #46307 — our GB10
>    (util 0.44, fixed `--kv-cache-memory-bytes`) had an EngineCore
>    abort mid-request at ~64k context on 09-02
>    (`CUBLAS_STATUS_INTERNAL_ERROR` in a bf16 GEMM during a DFlash
>    verify step; no Xid in dmesg, 550 W cap). Async CUDA reporting
>    means the faulting kernel may be elsewhere; the context-scaling
>    prefill working set you describe is a plausible source even at
>    conservative util.
>
> On the corruption root-cause (qo_len ≪ kv_len): in production we sit
> on exactly that configuration daily; we can contribute a
> deterministic repro ladder (greedy, fixed seeds, needle-verified)
> if that helps.

---

Nach Post: §8-Eintrag; falls Repro-Zusage → Task anlegen (Sweep-Wiederholung
mit gepatchtem FlashInfer auf sm120+sm121 = direkt unsere #53979-Validierung).
