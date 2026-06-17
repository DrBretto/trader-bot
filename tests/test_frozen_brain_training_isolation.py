"""PKT-TB-014 / D-AUTO-20260616: the monthly training + evolution jobs must
never read or overwrite the frozen ORB-1 New Brain weights. Isolation holds
structurally (the Batch image does not ship the frozen trees) and is enforced
by frozen_brain_guard at every training entrypoint."""
import re
from pathlib import Path

import pytest

from training.frozen_brain_guard import (
    assert_frozen_brain_isolated,
    FrozenBrainIsolationError,
    FROZEN_BRAIN_PATHS,
)

REPO = Path(__file__).resolve().parents[1]


def test_guard_passes_for_training_output_dirs():
    # The default train.py (/tmp/models) and evolve.py (/tmp/evolution) output
    # dirs are disjoint from the frozen brain tree.
    assert_frozen_brain_isolated('/tmp/models', '/tmp/evolution')


def test_guard_raises_when_output_under_frozen_tree():
    frozen = str(REPO / FROZEN_BRAIN_PATHS[1])  # models_out_007
    with pytest.raises(FrozenBrainIsolationError):
        assert_frozen_brain_isolated(frozen + '/regime')


def test_guard_raises_when_output_contains_frozen_path():
    with pytest.raises(FrozenBrainIsolationError):
        assert_frozen_brain_isolated(str(REPO / 'runs/pkt_tb_007_orthogonal_brain'))


def test_batch_training_image_excludes_frozen_brain():
    copy_lines = [
        l for l in (REPO / 'Dockerfile.training').read_text().splitlines()
        if l.strip().startswith(('COPY', 'ADD'))
    ]
    joined = "\n".join(copy_lines)
    assert not re.search(r'\bbrain/', joined), f"Dockerfile.training ships brain/: {copy_lines}"
    assert 'runs/' not in joined, f"Dockerfile.training ships runs/: {copy_lines}"
    assert 'models_out' not in joined, f"Dockerfile.training ships models_out: {copy_lines}"


def test_training_entrypoints_call_the_guard():
    assert 'assert_frozen_brain_isolated' in (REPO / 'training/train.py').read_text()
    assert 'assert_frozen_brain_isolated' in (REPO / 'evolution/evolve.py').read_text()
    assert 'assert_frozen_brain_isolated' in (REPO / 'automation/run_training_batch.sh').read_text()


def test_weekly_optimizer_installer_is_retired():
    installer = (REPO / 'scripts/install_optimizer_launchd.sh').read_text()
    assert 'RETIRED' in installer and 'exit 1' in installer
