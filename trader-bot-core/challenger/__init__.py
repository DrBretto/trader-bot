"""The INDEPENDENT challenger (PKT-TRADER-BOT-SEED-CANON-BY-REPLAY-V2 §challenger).

The blue-dotted challenger line = the LEGACY incumbent brain
(``decision_engine`` — health x regime ranker) run per day + the ORB-1 ``M1``
conviction tilt (<=8% NAV) overlaid, reconstructed INDEPENDENTLY of the two-stage
canon (it never reads the two-stage's intents / the coupled ``publish/challenger``).

Ported VERBATIM from the legacy sources into first-class ``trader-bot-core/``
paths (no runtime ``sys.path`` into ``runs/``):

  - ``incumbent.py``   <- ``src/steps/decision_engine.py`` (byte-identical)
  - ``genome.py``      <- ``runs/pkt_tb_007_orthogonal_brain/prototype/genome_007.py`` (byte-identical)
  - ``shadow_A.json``  <- ``runs/.../shadow/genomes/shadow_A.json`` (byte-identical)
  - ``tilt.py``        <- ``runs/.../prototype/tilt_adapter.py`` (sys.path hack removed; imports repointed)
  - ``lot_fix.py``     <- ``runs/.../prototype/lot_fix_007.py`` (Position import repointed)
  - ``risk_stats.py``  <- ``runs/.../prototype/risk_stats_007.py`` (OHLCV default repointed to the core store)
  - ``strategy.py``    <- ``src/utils/three_line_replay/strategies.py`` (StrategyContext + Strategy dataclasses)
  - ``book.py``        <- ``src/utils/three_line_replay/replay_engine.py`` (Position + Portfolio dataclasses)

The two-stage CANON is unchanged; only the challenger line is reconstructed here.
The one settled-close marking machinery (``replay.driver`` / ``Book.value``) marks
all three lines — the challenger differs only in the holdings it produces.
"""
