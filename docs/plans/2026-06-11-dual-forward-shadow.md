# 2026-06-11 — Dual Forward Shadow (PKT-TB-007 follow-on, operator-approved)

## Context

Operator approved the committee's ranked follow-on from PKT-TB-007: a costless dual forward
shadow. Two legs, both written nightly BEFORE outcomes exist: (1) paper shadow book(s) — the
incumbent's book with the brain's tilt applied on paper, marked nightly, rendered as a second
line on the dashboard next to the live line; (2) forecast-IC records — M1's nightly rankings,
scored at +5 trading days. Settles "is the forecast IC real forward" in ~7 months (Realist
spec); banks fresh holdout for any future brain. Configs: the value-blind a-priori genome
(shadow book A) and a-priori + M4-A damping (shadow book B, the pre-registered confirmation
arm). Verdict reads pre-registered in runs/pkt_tb_007_orthogonal_brain/shadow/SHADOW_PREREG.md.

## Plan

- [ ] Shadow engine under runs/pkt_tb_007_orthogonal_brain/shadow/ (imports prototype machinery):
      nightly catch-up-capable deterministic job — pull daily/<D> artifacts, run M1 (+M4-A) inference,
      compute both paper books via the tilt adapter, mark to close, append forecast records, score
      matured IC, persist state to S3 shadow/pkt_tb_007/ + local mirror, publish
      dashboard/shadow_timeseries.json
- [ ] SHADOW_PREREG.md — read dates, IC bar, utility equivalence band, M4-A sub-window rule (Realist spec)
- [ ] Tests (determinism, catch-up idempotence, no-look-ahead timestamps, S3 round-trip)
- [ ] launchd plist com.traderbot.shadow.plist (nightly, after the night pipeline lands)
- [ ] First night run end-to-end (manual), verify artifacts + JSON
- [ ] Frontend: render shadow line(s) when shadow_timeseries.json exists (graceful absent)
- [ ] STOP: ask operator before frontend deploy (CLAUDE.md checkpoint)

## Execution Log

- 2026-06-11: Branch ai/forward-shadow from ai/orthogonal-brain. Plan doc created.

## Follow-ups

- Read dates land automatically per SHADOW_PREREG (IC read ~31 weeks; utility reads at 8/14 months).
