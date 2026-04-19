"""
Unit tests for galfit_constraints.py - covers all six line grammars.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import galfit_constraints as gc


# -----------------------------------------------------------------------------
# Single-component soft forms
# -----------------------------------------------------------------------------

class TestSingleComponent:
    def test_absolute_range_with_to_keyword(self):
        entries = gc.read_constraints_text("3 n 0.7 to 5\n")
        assert len(entries) == 1
        e = entries[0]
        assert e.components == [3]
        assert e.coupling == "single"
        assert e.parameter == "n"
        assert e.bounds == (0.7, 5.0)
        assert e.absolute is True

    def test_relative_range_signed(self):
        entries = gc.read_constraints_text("2 x -1 0.5\n")
        e = entries[0]
        assert e.components == [2]
        assert e.parameter == "x"
        assert e.bounds == (-1.0, 0.5)
        assert e.absolute is False


# -----------------------------------------------------------------------------
# Hard-coupling forms
# -----------------------------------------------------------------------------

class TestHardCoupling:
    def test_offset_across_four_components(self):
        e = gc.read_constraints_text("3_2_1_9 x offset\n")[0]
        assert e.components == [3, 2, 1, 9]
        assert e.coupling == "offset"
        assert e.parameter == "x"
        assert e.bounds == "offset"

    def test_ratio_lock(self):
        e = gc.read_constraints_text("1_5_3_2 re ratio\n")[0]
        assert e.components == [1, 5, 3, 2]
        assert e.coupling == "ratio"
        assert e.parameter == "re"
        assert e.bounds == "ratio"


# -----------------------------------------------------------------------------
# Pairwise forms
# -----------------------------------------------------------------------------

class TestPairwise:
    def test_diff_separator(self):
        e = gc.read_constraints_text("3-7 mag -0.5 3\n")[0]
        assert e.components == [3, 7]
        assert e.coupling == "diff"
        assert e.bounds == (-0.5, 3.0)

    def test_ratio_bounds_separator(self):
        e = gc.read_constraints_text("3/5 re 1 3\n")[0]
        assert e.components == [3, 5]
        assert e.coupling == "ratio_bounds"
        assert e.bounds == (1.0, 3.0)


# -----------------------------------------------------------------------------
# Parameter-name vocabulary
# -----------------------------------------------------------------------------

class TestParamNames:
    @pytest.mark.parametrize("name", [
        "x", "y", "mag", "re", "rs", "n", "alpha", "beta", "gamma",
        "pa", "q", "c", "f1a", "f1p", "f6a", "b2", "r5", "r10",
    ])
    def test_documented_names_pass(self, name):
        assert gc.is_known_param_name(name)

    def test_integer_names_accepted(self):
        assert gc.is_known_param_name("9")

    def test_unknown_name_parses_anyway(self):
        # Validation is advisory; unknown names must still round-trip.
        e = gc.read_constraints_text("2 wiggle 0 1\n")[0]
        assert e.parameter == "wiggle"


# -----------------------------------------------------------------------------
# Example file (the one shipped in the GALFIT install)
# -----------------------------------------------------------------------------

def test_parses_local_example_constraints():
    path = Path("/Users/shuang/code/galfit/EXAMPLE.CONSTRAINTS")
    if not path.exists():
        pytest.skip("EXAMPLE.CONSTRAINTS not present")
    entries = gc.read_constraints(path)
    # The example has exactly 6 lines (one per documented grammar form).
    assert len(entries) == 6
    kinds = [e.coupling for e in entries]
    assert "offset" in kinds
    assert "ratio" in kinds
    assert "single" in kinds
    assert "diff" in kinds
    assert "ratio_bounds" in kinds


# -----------------------------------------------------------------------------
# Round-trip
# -----------------------------------------------------------------------------

@pytest.fixture
def mixed_entries():
    text = """\
# My constraints
3 n 0.7 to 5
2 x -1 0.5
3_2_1_9 x offset
1_5_3_2 re ratio
3-7 mag -0.5 3
3/5 re 1 3
"""
    return text, gc.read_constraints_text(text)


class TestRoundTrip:
    def test_text_round_trip(self, mixed_entries):
        _, entries = mixed_entries
        text = gc.write_constraints_text(entries)
        reparsed = gc.read_constraints_text(text)
        assert gc.entries_to_dicts(reparsed) == gc.entries_to_dicts(entries)

    def test_yaml_round_trip(self, mixed_entries):
        _, entries = mixed_entries
        y = gc.to_yaml_constraints(entries)
        reparsed = gc.from_yaml_constraints(y)
        assert gc.entries_to_dicts(reparsed) == gc.entries_to_dicts(entries)

    def test_json_round_trip(self, mixed_entries):
        _, entries = mixed_entries
        j = gc.to_json_constraints(entries)
        reparsed = gc.from_json_constraints(j)
        assert gc.entries_to_dicts(reparsed) == gc.entries_to_dicts(entries)
