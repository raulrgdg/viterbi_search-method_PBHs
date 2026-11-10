"""v1.3 options: several detectors, chunk high-pass and per-SFT normalisation (studies/multi_detector)."""

import numpy as np

import pytest

from pbh_viterbi.paths import noise_background_path
from pbh_viterbi.sft.maps import network_track_product, track_product
from pbh_viterbi.workflows.common import fast_products


def _sft(seed, n_bins=200, n_sft=120):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n_bins, n_sft)) + 1j * rng.normal(size=(n_bins, n_sft))


def test_one_map_equals_the_single_detector_product():
    sft = _sft(1)
    one, net = track_product(sft, 10.0), network_track_product([sft], 10.0)
    np.testing.assert_allclose(net["power"], one["power"])
    np.testing.assert_array_equal(net["track_index"], one["track_index"])


def test_summed_map_is_the_scaled_sum_of_the_normalised_maps():
    a, b = _sft(2), _sft(3) * 7.0  # a different noise level is removed by each map's normalisation
    net = network_track_product([a, b], 10.0)
    expected = (track_product(a, 10.0)["power"] + track_product(_sft(3), 10.0)["power"]) / np.sqrt(2)
    np.testing.assert_allclose(net["power"], expected)


def test_one_detector_dict_equals_the_single_detector_path():
    rng = np.random.default_rng(4)
    from pycbc.types import TimeSeries

    segments = [TimeSeries(rng.normal(size=512 * 64), delta_t=1 / 512) for _ in range(2)]
    single = fast_products(segments, [8.0])
    as_dict = fast_products({"H1": segments}, [8.0])
    np.testing.assert_array_equal(single[0]["power"], as_dict[0]["power"])


def test_background_file_names():
    assert noise_background_path("95").name == "noise_background_tsft95_chunkhp_sftnorm.csv"  # v1.3 default
    assert noise_background_path("95", ["H1"], "segment", "bin").name == "noise_background_tsft95.csv"  # v1.2
    assert noise_background_path("95", ["H1", "L1"], "segment", "bin").name == "noise_background_tsft95_H1L1.csv"
    assert (noise_background_path("95", ["H1", "L1"], "chunk", "bin+sft").name
            == "noise_background_tsft95_H1L1_chunkhp_sftnorm.csv")


def test_chunk_highpass_filters_the_strain_once():
    from pycbc.types import TimeSeries

    from pbh_viterbi.sft.fast import highpass, sft_matrix

    rng = np.random.default_rng(5)
    x = rng.normal(size=512 * 64)
    segments = [TimeSeries(x, delta_t=1 / 512)]
    chunk = fast_products(segments, [8.0], sft_highpass="chunk")[0]
    expected = track_product(sft_matrix(highpass(x, 512, 61.1), 512, 8.0, 61.1, 65.7, prefiltered=True), 8.0)
    np.testing.assert_array_equal(chunk["power"], expected["power"])
    np.testing.assert_array_equal(fast_products(segments, [8.0])[0]["power"], chunk["power"])  # default
    segment = fast_products(segments, [8.0], sft_highpass="segment")[0]
    assert not np.allclose(chunk["power"], segment["power"])
    assert (noise_background_path("95", ["H1", "L1"], "chunk", "bin").name
            == "noise_background_tsft95_H1L1_chunkhp.csv")


def test_signal_list(tmp_path):
    from pbh_viterbi.workflows.injected_search import read_signal_list

    path = tmp_path / "signals.csv"
    path.write_text("signal_id,mchirp,distance\n1000,0.1,0.2\n1001,0.0004,0.001\n")
    signals = read_signal_list(path)
    assert sorted(signals) == [1000, 1001]
    m1, m2, d = signals[1000]
    assert m1 == m2 and d == 0.2
    from pycbc.conversions import mchirp_from_mass1_mass2

    assert abs(mchirp_from_mass1_mass2(m1, m2) - 0.1) < 1e-9


def test_sft_normalisation_flattens_a_broadband_glitch_and_keeps_a_line():
    from pbh_viterbi.sft.maps import normalized_power

    sft = _sft(5, n_bins=400, n_sft=200)
    sft[:, 50] *= 30.0          # broadband glitch: whole band up in one SFT
    sft[123, 100:110] *= 6.0    # narrow signal: one bin, ten SFTs
    plain = normalized_power(sft, "bin")
    flat = normalized_power(sft, "bin+sft")
    assert np.median(plain[50]) > 20 * np.median(plain[49])
    np.testing.assert_allclose(np.median(flat[50]), np.median(flat[49]), rtol=0.15)
    np.testing.assert_allclose(flat[100:110, 123], plain[100:110, 123], rtol=0.1)
    np.testing.assert_allclose(np.median(flat), np.median(plain), rtol=0.02)  # same overall scale


def test_unknown_normalisation_is_rejected():
    import pytest

    from pbh_viterbi.sft.maps import normalized_power

    with pytest.raises(ValueError):
        normalized_power(_sft(6), "sft")


def test_zero_padded_sft_is_an_error_with_bin_sft():
    from pbh_viterbi.sft.maps import normalized_power

    sft = _sft(7)
    sft[:, 10:20] = 0  # a zero-padded frame
    normalized_power(sft, "bin")  # still defined per bin
    with pytest.raises(ValueError, match="zero median"):
        normalized_power(sft, "bin+sft")


def test_non_finite_strain_is_an_error():
    from pbh_viterbi.sft.maps import normalized_power

    sft = _sft(8)
    sft[3, 40] = np.nan
    for mode in ("bin", "bin+sft"):
        with pytest.raises(ValueError, match="non-finite"):
            normalized_power(sft, mode)


def test_signal_ids_must_be_in_the_signal_list(tmp_path, monkeypatch):
    from pbh_viterbi.workflows import injected_search

    path = tmp_path / "signals.csv"
    path.write_text("signal_id,mchirp,distance\n1000,0.1,0.2\n")
    for extra, message in ((["--signal-ids", "1000,1001"], "not in --signal-list"),
                           (["--signal-ids", ","], "empty"),
                           (["--signal-ids", "a"], "integers"),
                           (["--signals", "0:1"], "cannot be combined")):
        monkeypatch.setattr("sys.argv", ["injected_search", "--pack", "3", "--signal-list", str(path), *extra,
                                         "--noise-background", str(noise_background_path("95", ["H1"], "segment", "bin"))])
        with pytest.raises(ValueError, match=message):
            injected_search.main()


def test_duplicate_signal_id_is_an_error(tmp_path):
    from pbh_viterbi.workflows.injected_search import read_signal_list

    path = tmp_path / "signals.csv"
    path.write_text("signal_id,mchirp,distance\n1000,0.1,0.2\n1000,0.01,0.1\n")
    with pytest.raises(ValueError, match="twice"):
        read_signal_list(path)


def test_makesfts_needs_the_segment_highpass_and_a_background_must_exist(tmp_path):
    from argparse import Namespace

    from pbh_viterbi.workflows.common import resolve_tsft_and_background

    def args(**kw):
        base = dict(tsft=[8.0], tsft_list="95", detectors=["H1"], sft_method="fast", sft_highpass="chunk",
                    map_normalisation="bin+sft", noise_background=None, mode="search")
        return Namespace(**{**base, **kw})

    with pytest.raises(ValueError, match="--sft-highpass segment"):
        resolve_tsft_and_background(args(sft_method="makesfts"))
    with pytest.raises(FileNotFoundError, match="nsigma background"):
        resolve_tsft_and_background(args(noise_background=tmp_path / "missing.csv"))
    resolve_tsft_and_background(args(noise_background=tmp_path / "missing.csv", mode="background"))  # not read
