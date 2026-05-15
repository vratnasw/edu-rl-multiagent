"""Smoke test: imports all module stubs and verifies they raise
NotImplementedError at run() time. Phase A scaffold guarantee."""
from __future__ import annotations

import sys
from pathlib import Path

# Make src/ importable without an installed package
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def test_imports_and_not_implemented():
    import pytest
    from src.environment import hierarchical_env as _environment_hierarchical_env
    from src.agents import state_agent as _agents_state_agent
    from src.agents import county_agent as _agents_county_agent
    from src.agents import district_agent as _agents_district_agent
    from src.communication import message_passing as _communication_message_passing
    from src.analysis import stackelberg_analysis as _analysis_stackelberg_analysis
    mods = [_environment_hierarchical_env, _agents_state_agent, _agents_county_agent, _agents_district_agent, _communication_message_passing, _analysis_stackelberg_analysis]
    assert len(mods) > 0, "no modules collected"
    for m in mods:
        with pytest.raises(NotImplementedError):
            m.run()


def test_config_loads():
    sys.path.insert(0, str(REPO / "src"))
    from utils.config_loader import load_config
    cfg = load_config()
    assert isinstance(cfg, dict)
    assert "data" in cfg
