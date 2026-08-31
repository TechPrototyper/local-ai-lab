# Draft response to jethac's #53979 review (SWA + non-causal finding)

*Prepared 2026-08-29 during the night watch. NOT posted / NOT committed —
this touches our attested PR under Tim's name, so it is Tim's call. Ready
to fire on his go.*

## The finding (jethac, 00:52 UTC)

`plan(causal=False, window_left>=0)` on the FA2-NVFP4 non-causal wrapper is
an unvalidated combination: FlashInfer applies the sliding window in-kernel
regardless of mask mode, so it yields an end-aligned left-only staircase
with no right clip. **Not reachable by our validated arms** (Qwen3.8,
full-attention drafter, `window_left=-1`) but reachable by the
`use_swa=True`/`layer_types=None` DFlash family (e.g. MiMo-V2.5-Pro-FP4-
DFlash). Two accepted fixes: (1) `raise` until a kernel-parity test exists,
or (2) carry a symmetric window in a custom mask like the grouped path.

## Recommendation: Option 1 (guard) now, Option 2 as a follow-up

Option 1 is minimal, unblocks merge, and keeps every validated config
byte-identical (window_left=-1 passes straight through). Option 2 is real
kernel work and wants a parity test first — better as its own PR so it
carries its own attestation and validation, exactly the line's rule.

## The guard (exact placement)

`vllm/v1/attention/backends/flashinfer.py`, in the `else:` branch that
already selects the non-causal FA2-NVFP4 wrapper (~line 1414, where
`noncausal_backend = "fa2" if self.use_fa2_nvfp4_kv else "auto"`). We are
already inside the non-causal path, so jethac's `not causal` is implied and
the check reduces to `use_fa2_nvfp4_kv and window_left >= 0`:

```python
noncausal_backend = "fa2" if self.use_fa2_nvfp4_kv else "auto"
if noncausal_backend == "fa2" and self.window_left is not None and self.window_left >= 0:
    # Non-causal + sliding window is unvalidated on the FA2 NVFP4-KV path:
    # FlashInfer applies window_left in-kernel regardless of mask mode, so
    # causal=False + window_left>=0 is a left-only staircase no parity test
    # has covered. Raise until one does (see #53979 review, jethac). The
    # validated arms use window_left=-1 and are unaffected.
    raise NotImplementedError(
        "FA2 NVFP4-KV non-causal prefill with a sliding window "
        "(window_left >= 0) is not yet validated on consumer Blackwell; "
        "blocked pending a kernel-parity test for the causal=False + "
        "window_left>=0 mode."
    )
```

*Verify before committing:* the attribute name (`self.window_left`) and its
"no window" sentinel (`-1`) — grep confirmed `window_left=-1` is the
no-window value our arms use; the sink-wrapper branch just above passes
`self.window_left` straight to the sink wrapper, so the attribute exists on
`self` at this point.

## Draft reply comment (for #53979 / the #46329 thread)

> Thanks — that's the right catch, and I'd rather block it than ship a mode
> I haven't put paired greedy numbers behind. Taking **Option 1**: guarding
> `use_fa2_nvfp4_kv and not causal and window_left >= 0` with a
> `NotImplementedError` until a kernel-parity test covers the kNone+window
> staircase. It leaves the validated arms (`window_left=-1`) byte-identical
> and keeps the merge honest about what's actually been exercised. The
> symmetric-mask route (Option 2) is the better long-term fix but it's real
> kernel work with its own parity test — I'd rather it land as its own PR so
> it carries its own attestation and validation, same as this line's rule.
> Pushing the guard shortly; the MiMo-DFlash SWA family is the config that
> would want Option 2 when someone validates it.
>
> And noted on the composition with #46443 / the mm-prefix seam — agreed
> that's the grouping's job, not this delta's. Thanks for the thorough read.

## After posting (if Tim approves the code path)

1. Add the guard on the fork branch `integrate/dflash2-nvfp4-v4` (or the
   #53979 source branch), commit `Co-authored-by` NOT needed (our code).
2. No GPU revalidation strictly required — the guard only *adds* a raise on
   an untested path; the validated arms don't hit it. A quick CPU
   import/guard-path probe (like the #41 lmhead guard probe) would confirm
   it raises exactly on `(fa2, non-causal, window_left>=0)` and passes on
   `window_left=-1`. Cheap, and matches the "prove the guard" habit.
3. Push; the review finding is then resolved.
