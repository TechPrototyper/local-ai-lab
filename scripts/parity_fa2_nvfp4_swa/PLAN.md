# Kernel-Parity-Probe: non-causal + SWA auf dem FA2-NVFP4-Pfad — SCAFFOLD

**Status: Scaffold / Vorbereitung. KEIN Verdikt.** (2026-09-01, Nacht)

## Warum
Der SWA-Guard (unser Beitrag, #53979) hält **NVFP4-KV + DFlash2** getrennt:
der DFlash2-Drafter ist all-layers sliding-window (`sliding_window: 2048`), und
**non-causal + window auf dem FA2-NVFP4-Pfad hat noch keinen Kernel-Parity-Test**
(jethacs Finding). Die Kombination *serviert* auf sm121 byte-identisch greedy —
starke Empirie —, aber die formale Probe ist die Upstream-Zusage. Bis sie existiert:
`turbo = fp8-KV`, `max-pool = nvfp4-KV no-spec`. **Diese Probe zu schreiben ist
"the next work item".**

## Werkzeug (vorhanden, lokal, CPU)
- `results/byteorder_cpu_proof.py` — spiegelt bereits vLLMs
  `dequant_nvfp4_kv_cache` (tests/kernels/quantization/nvfp4_utils.py) und beweist
  Shim-vs-real-page-Agreement OHNE GPU (reines torch 2.7 CPU auf dem Mac).
- `results/RESULT_nvfp4_byteorder_verdict.json` — belegt: vLLM speichert Block-
  Scales 4x4-SWIZZLED; linear gelesen ⇒ falsche Drafter-KV. Das ist die
  Grundlage, auf der die SWA/non-causal-Parity aufsetzt.
- jethac `dequant_nvfp4_kv_cache(..., swizzle_sf=...)` @ `7a5cf143` (Branch
  `nvfp4-fa2-consumer-blackwell`) — der Reference-Reader für die GPU-Seite.

## Was die Probe zeigen muss
Für die non-causal + sliding-window(2048)-Maske auf dem FA2-NVFP4-Pfad:
1. **Referenz** (dequant → dichte Attention in fp32/bf16, mit derselben SWA-Maske
   und non-causal-Struktur) vs.
2. **Kernel** (FA2-NVFP4 mit `use_trtllm_attention:false`, wie Prod) —
   Ausgaben `torch.testing.assert_close` bei realistischer Toleranz, über
   mehrere Head-Sizes (128/256) und Fensterlagen (innerhalb/über der 2048-Grenze).

## TODO (morgen, supervised — GPU nötig für den Kernel-Arm)
- [ ] `probe_skeleton.py` ausbauen: Referenz-Arm (CPU, aus byteorder_cpu_proof
      ableiten) + Kernel-Arm-Hook (GPU, jethacs Reader). **Derzeit nur CPU-Referenz
      + Stub für den Kernel-Arm — bewusst kein erfundenes Ergebnis.**
- [ ] Testvektoren: non-causal Q/K/V, SWA-Maske 2048, Sink-Outlier ~125k einbauen.
- [ ] `assert_close`-Toleranzen mit jethac abstimmen (Review-Grade).
- [ ] Ergebnis → #53979-Thread (NUR nach Tims Go; Publizieren manuell).

## Abgrenzung
Kein Solo-Verdikt. Der Kernel-Arm braucht GPU + jethac-Abstimmung; das Ergebnis
ist Review-relevant für Upstream. Scaffold liefert die CPU-Referenz-Hälfte + die
Struktur, damit der betreute Lauf morgen nur noch den Kernel-Arm anhängt.
