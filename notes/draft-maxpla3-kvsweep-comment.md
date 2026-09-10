# ENTWURF v2 (Spock-Ton, Tim 09-02: trocken, keine Bewertung, Zahlen/Daten/
# Fakten, freundlicher Gruß) — Posten nach erfolgreichem Sweep autorisiert.
# Platzhalter {…} aus results/kvsweep-rtx-v5.json füllen.

---

> Hi maxpla3 — data point from our side on the same axis.
>
> Setup: RTX 5090, this PR's line rebased to `9590d2464` plus seam fixes
> (#53977/#53979) and #50897, serving a 13 GB product-codebook NVFP4
> checkpoint, `--kv-cache-dtype nvfp4`, MTP with `num_speculative_tokens=2`,
> KV pool sized to ~775k tokens. Same measurement shape as yours: 8 runs per
> context size, decode rate from stream timing, contexts 1k→262k.
>
> Results (decode tok/s, 8-run means):
> {TABELLE: ctx / decode_mean / accept_len}
>
> Mechanism, as observed in our metrics: each speculative step drafts k
> tokens whose KV occupies pool blocks before verification; at high pool
> fill this displaces reusable prefix blocks, and the displacement rate
> scales with k. Acceptance length in our data changes from {A} (1k ctx) to
> {B} (262k ctx). At k=2 the decode curve flattens to {X}% of its 1k value
> at 262k; measured at k=3 on the pre-rebase head our drop was {Y}%,
> consistent with the shape of your plot.
>
> Note on baselines: `7a5cf1431` predates the rebase; the current head is
> `9590d2464`.
>
> Methodology and full data: {Notebook-Link}. Happy measuring — greetings
> from another 5090 in production.

---

Nach dem Post: §8-Watch-Log-Eintrag. Anmerkung: den {Y}-Vergleichsarm (k=3)
nur aufnehmen, wenn er tatsächlich gemessen wird — sonst Satz streichen,
nichts behaupten, was nicht gemessen ist.
