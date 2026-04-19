"""
Unit tests for galfit_io.py

Run with: pytest tests/test_galfit_io.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

from dataclasses import fields

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import galfit_io as gio

FIXTURES = Path(__file__).parent / "fixtures" / "galfit"


# -----------------------------------------------------------------------------
# Low-level helpers
# -----------------------------------------------------------------------------

class TestEnvelopeStripping:
    def test_bare(self):
        inner, fixed, constr, susp = gio._strip_envelope("4.30")
        assert inner == "4.30"
        assert not (fixed or constr or susp)

    def test_square_brackets(self):
        inner, fixed, constr, susp = gio._strip_envelope("[4.30]")
        assert inner == "4.30"
        assert fixed and not constr and not susp

    def test_curly_braces(self):
        inner, fixed, constr, susp = gio._strip_envelope("{4.30}")
        assert inner == "4.30"
        assert constr and not fixed and not susp

    def test_stars(self):
        inner, fixed, constr, susp = gio._strip_envelope("*0.15*")
        assert inner == "0.15"
        assert susp and not fixed and not constr

    def test_nested_bracket_star(self):
        inner, fixed, constr, susp = gio._strip_envelope("[*0.15*]")
        assert inner == "0.15"
        assert fixed and susp and not constr

    def test_negative_value(self):
        inner, *_ = gio._strip_envelope("-2.5")
        assert inner == "-2.5"

    def test_scientific(self):
        inner, *_ = gio._strip_envelope("1.2e-3")
        assert gio._parse_number(inner) == pytest.approx(1.2e-3)


class TestTokenizer:
    def test_gather_parenthesized(self):
        toks = gio._gather_value_tokens("48.5 51.3 (0.004 0.004) 1 1")
        assert toks == ["48.5", "51.3", "(0.004 0.004)", "1", "1"]

    def test_inline_single_paren(self):
        toks = gio._gather_value_tokens("20.089 (0.012) 1")
        assert toks == ["20.089", "(0.012)", "1"]

    def test_comment_stripping(self):
        body, comment = gio._split_comment("3) 20.0   1 # total magnitude")
        assert body.strip().startswith("3)")
        assert "total magnitude" in comment

    def test_key_line_regex(self):
        m = gio._KEY_LINE_RE.match("T0) radial2-b")
        assert m is not None
        assert m.group(1) == "T0"


class TestConsumeFittedValues:
    def test_one_value_input(self):
        fvs, rest = gio._consume_fitted_values(["4.30", "1"], 1)
        assert len(fvs) == 1
        assert fvs[0].value == pytest.approx(4.30)
        assert fvs[0].fit is True
        assert fvs[0].uncertainty is None
        assert rest == []

    def test_one_value_with_err(self):
        fvs, _ = gio._consume_fitted_values(["20.089", "(0.012)", "1"], 1)
        assert fvs[0].value == pytest.approx(20.089)
        assert fvs[0].uncertainty == pytest.approx(0.012)
        assert fvs[0].fit is True

    def test_two_value_input(self):
        fvs, _ = gio._consume_fitted_values(["48.5", "51.3", "1", "1"], 2)
        assert len(fvs) == 2
        assert fvs[0].value == pytest.approx(48.5)
        assert fvs[1].value == pytest.approx(51.3)

    def test_two_value_with_err(self):
        fvs, _ = gio._consume_fitted_values(
            ["48.5", "51.3", "(0.004 0.004)", "1", "1"], 2,
        )
        assert fvs[0].uncertainty == pytest.approx(0.004)
        assert fvs[1].uncertainty == pytest.approx(0.004)

    def test_envelopes_propagate(self):
        fvs, _ = gio._consume_fitted_values(["[4.0]", "0"], 1)
        assert fvs[0].value == pytest.approx(4.0)
        assert fvs[0].fixed_in_output
        assert fvs[0].fit is False


# -----------------------------------------------------------------------------
# Header parsing
# -----------------------------------------------------------------------------

class TestHeader:
    def test_simple_round_trip(self):
        gf = gio.read_galfit(FIXTURES / "simple_sersic.galfit")
        h = gf.header
        assert h.input_image == "gal.fits"
        assert h.output_block == "imgblock.fits"
        assert h.sigma_image == "none"
        assert h.psf_image == "psf.fits"
        assert h.psf_diffusion_kernel is None
        assert h.psf_fine_sampling == 1
        assert h.fit_region == (1, 93, 1, 93)
        assert h.conv_box == (100, 100)
        assert h.zeropoint == pytest.approx(26.563)
        assert h.plate_scale == pytest.approx((0.038, 0.038))
        assert h.display_type == "regular"
        assert h.options == 0

    def test_psf_with_kernel(self):
        gf = gio.read_galfit(FIXTURES / "all_profiles.galfit")
        assert gf.header.psf_image == "psf.fits"
        assert gf.header.psf_diffusion_kernel == "kernel.txt"
        assert gf.header.psf_fine_sampling == 2
        assert gf.header.options == 3


# -----------------------------------------------------------------------------
# Profile parsing
# -----------------------------------------------------------------------------

class TestProfiles:
    @pytest.fixture(scope="class")
    def all_profiles(self):
        return gio.read_galfit(FIXTURES / "all_profiles.galfit")

    def test_count(self, all_profiles):
        # 10 light + 1 sky
        assert len(all_profiles.components) == 11

    def test_sersic_with_flux_norm_suffix(self, all_profiles):
        c = all_profiles.components[0]
        assert isinstance(c, gio.SersicComponent)
        assert c.flux_norm_suffix == 2
        assert c.profile_token == "sersic2"
        assert c.mag.value == pytest.approx(18.5)
        assert c.n.value == pytest.approx(4.0)

    def test_nuker_has_alpha_beta_gamma(self, all_profiles):
        c = next(c for c in all_profiles.components
                 if isinstance(c, gio.NukerComponent))
        assert c.alpha.value == pytest.approx(1.2)
        assert c.beta.value == pytest.approx(0.5)
        assert c.gamma.value == pytest.approx(0.7)

    def test_king_has_rc_rt(self, all_profiles):
        c = next(c for c in all_profiles.components
                 if isinstance(c, gio.KingComponent))
        assert c.rc.value == pytest.approx(5.0)
        assert c.rt.value == pytest.approx(30.0)

    def test_edgedisk_has_no_axis_ratio(self, all_profiles):
        c = next(c for c in all_profiles.components
                 if isinstance(c, gio.EdgediskComponent))
        assert not hasattr(c, "q")
        assert c.h_s.value == pytest.approx(2.5)
        assert c.r_s.value == pytest.approx(8.0)

    def test_psf_only_has_position_and_mag(self, all_profiles):
        c = next(c for c in all_profiles.components
                 if isinstance(c, gio.PsfComponent))
        assert c.mag.value == pytest.approx(18.2)
        assert not hasattr(c, "re")

    def test_sky_uses_dc_not_xy(self, all_profiles):
        c = next(c for c in all_profiles.components
                 if isinstance(c, gio.SkyComponent))
        assert c.dc.value == pytest.approx(0.5)
        assert c.dc.fit is False


# -----------------------------------------------------------------------------
# Hidden blocks
# -----------------------------------------------------------------------------

class TestHiddenBlocks:
    @pytest.fixture(scope="class")
    def hidden(self):
        return gio.read_galfit(FIXTURES / "hidden_blocks.galfit")

    def test_component_count(self, hidden):
        # 1 sersic + 2 truncation + 1 sky
        assert len(hidden.components) == 4

    def test_c0_parsed(self, hidden):
        c = hidden.components[0]
        assert c.hidden.c0 is not None
        assert c.hidden.c0.value == pytest.approx(0.2)

    def test_bending_sparse(self, hidden):
        bending = hidden.components[0].hidden.bending
        assert set(bending.keys()) == {1, 3}
        assert bending[1].value == pytest.approx(0.05)
        assert bending[3].value == pytest.approx(0.02)

    def test_fourier_sparse_indices(self, hidden):
        fourier = hidden.components[0].hidden.fourier
        assert set(fourier.keys()) == {1, 6, 20}
        amp_1, phase_1 = fourier[1]
        assert amp_1.value == pytest.approx(0.07)
        assert phase_1.value == pytest.approx(30.0)

    def test_rotation_powerlaw(self, hidden):
        rot = hidden.components[0].hidden.rotation
        assert rot.mode == "powerlaw"
        assert rot.r_in.value == pytest.approx(30.0)
        assert rot.r_out.value == pytest.approx(100.0)
        assert rot.theta_out.value == pytest.approx(275.0)
        assert rot.incl.value == pytest.approx(0.5)
        assert rot.sky_pa.value == pytest.approx(30.0)

    def test_trunc_refs(self, hidden):
        hb = hidden.components[0].hidden
        assert hb.trunc_inner_ref == 3
        assert hb.trunc_outer_ref == 2

    def test_truncation_component_block(self, hidden):
        tc1 = hidden.components[1]
        assert isinstance(tc1, gio.TruncationComponent)
        assert tc1.trunc_type == "radial"
        assert tc1.t4.value == pytest.approx(4.4)
        assert tc1.t5.value == pytest.approx(9.2)
        # Attached Fourier mode on the truncation block:
        assert 1 in tc1.hidden.fourier
        assert tc1.hidden.fourier[1][0].value == pytest.approx(0.1)

    def test_truncation_type_2b(self, hidden):
        tc2 = hidden.components[2]
        assert isinstance(tc2, gio.TruncationComponent)
        assert tc2.trunc_type == "radial2-b"
        assert tc2.t2.value == pytest.approx(50.0)
        assert tc2.t3.value == pytest.approx(70.0)


# -----------------------------------------------------------------------------
# Output decorations
# -----------------------------------------------------------------------------

class TestOutputDecorations:
    @pytest.fixture(scope="class")
    def decorated(self):
        return gio.read_galfit(FIXTURES / "output_decorations.galfit")

    def test_leading_comments_preserved(self, decorated):
        assert any("Chi^2" in c for c in decorated.leading_comments)

    def test_position_uncertainty(self, decorated):
        c = decorated.components[0]
        assert c.x.uncertainty == pytest.approx(0.0042)
        assert c.y.uncertainty == pytest.approx(0.0039)

    def test_magnitude_uncertainty(self, decorated):
        c = decorated.components[0]
        assert c.mag.value == pytest.approx(20.089)
        assert c.mag.uncertainty == pytest.approx(0.0123)

    def test_star_wrapped(self, decorated):
        c = decorated.components[0]
        assert c.re.suspicious is True
        assert c.re.value == pytest.approx(0.32)

    def test_bracket_wrapped_sets_fixed(self, decorated):
        c = decorated.components[0]
        assert c.n.fixed_in_output is True
        assert c.n.fit is False
        assert c.n.value == pytest.approx(4.0)

    def test_curly_wrapped_sets_constrained(self, decorated):
        c = decorated.components[0]
        assert c.q.constrained_in_output is True
        assert c.q.value == pytest.approx(0.3)


# -----------------------------------------------------------------------------
# Round-trip
# -----------------------------------------------------------------------------

class TestRoundTrip:
    @pytest.mark.parametrize("fixture_name", [
        "simple_sersic.galfit",
        "all_profiles.galfit",
        "hidden_blocks.galfit",
    ])
    def test_text_to_text_semantic(self, fixture_name):
        original = gio.read_galfit(FIXTURES / fixture_name)
        text = gio.write_galfit_text(original)
        reparsed = gio.read_galfit_text(text)
        # Compare via dict representation (stable ordering by construction).
        assert gio.to_dict(reparsed) == gio.to_dict(original)

    @pytest.mark.parametrize("fixture_name", [
        "simple_sersic.galfit",
        "all_profiles.galfit",
        "hidden_blocks.galfit",
        "output_decorations.galfit",
    ])
    def test_yaml_round_trip(self, fixture_name):
        original = gio.read_galfit(FIXTURES / fixture_name)
        text = gio.to_yaml(original)
        reparsed = gio.from_yaml(text)
        assert gio.to_dict(reparsed) == gio.to_dict(original)

    @pytest.mark.parametrize("fixture_name", [
        "simple_sersic.galfit",
        "all_profiles.galfit",
        "hidden_blocks.galfit",
        "output_decorations.galfit",
    ])
    def test_json_round_trip(self, fixture_name):
        original = gio.read_galfit(FIXTURES / fixture_name)
        text = gio.to_json(original)
        reparsed = gio.from_json(text)
        assert gio.to_dict(reparsed) == gio.to_dict(original)

    def test_write_strips_decorations_by_default(self):
        original = gio.read_galfit(FIXTURES / "output_decorations.galfit")
        text = gio.write_galfit_text(original, preserve_decorations=False)
        reparsed = gio.read_galfit_text(text)
        # Every FittedValue in the reparsed file has its decoration fields
        # cleared (uncertainty dropped, envelopes normalized away).
        for comp in reparsed.components:
            for f in fields(comp):
                v = getattr(comp, f.name)
                if isinstance(v, gio.FittedValue):
                    assert v.uncertainty is None
                    assert v.fixed_in_output is False
                    assert v.constrained_in_output is False
                    assert v.suspicious is False

    def test_write_preserves_decorations_when_requested(self):
        original = gio.read_galfit(FIXTURES / "output_decorations.galfit")
        text = gio.write_galfit_text(original, preserve_decorations=True)
        assert "(" in text and ")" in text
        assert "[" in text and "]" in text
        assert "{" in text and "}" in text
        assert "*" in text


# -----------------------------------------------------------------------------
# Legacy pre-3.0 mapping (warn but accept)
# -----------------------------------------------------------------------------

class TestLegacy:
    def test_parameter_8_maps_to_q_with_warning(self, caplog):
        text = """\
A) gal.fits
B) out.fits
C) none
D) psf.fits
E) 1
F) none
G) none
H) 1 100 1 100
I) 50 50
J) 27.0
K) 0.1 0.1
O) regular
P) 0

 0) sersic
 1) 50 50 1 1
 3) 20 1
 4) 5 1
 5) 4 1
 8) 0.5 1
10) 0 1
 Z) 0
"""
        with caplog.at_level("WARNING", logger="galfit_io"):
            gf = gio.read_galfit_text(text)
        assert any("pre-3.0" in r.message for r in caplog.records)
        assert gf.components[0].q.value == pytest.approx(0.5)
