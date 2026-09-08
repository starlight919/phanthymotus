import numpy as np
import pytest

from plugins.matcha_phonetone.trt_runtime import (
    fix_len_compatibility,
    generate_path,
    intersperse,
    regulate_encoder,
)


def test_intersperse_adds_matcha_blank_boundaries():
    assert intersperse((3, 7, 11)).tolist() == [0, 3, 0, 7, 0, 11, 0]


def test_fix_len_compatibility_rounds_up_to_unet_factor():
    assert fix_len_compatibility(1) == 4
    assert fix_len_compatibility(4) == 4
    assert fix_len_compatibility(5) == 8


def test_generate_path_uses_cumulative_duration_boundaries():
    path = generate_path(
        np.asarray([[2, 1, 3]], dtype=np.float32),
        np.ones((1, 3, 6), dtype=np.float32),
    )
    assert path[0].argmax(axis=0).tolist() == [0, 0, 1, 2, 2, 2]


def test_regulate_encoder_matches_ceil_then_length_scale_semantics():
    regulated = regulate_encoder(
        mu_x=np.asarray([[[10.0, 20.0], [1.0, 2.0]]], dtype=np.float32),
        logw=np.log(np.asarray([[[1.1, 2.0]]], dtype=np.float32)),
        x_mask=np.ones((1, 1, 2), dtype=np.float32),
        length_scale=0.5,
    )
    assert regulated.lengths.tolist() == [2]
    assert regulated.valid_frames == 2
    assert regulated.mask.shape == (1, 1, 4)
    np.testing.assert_array_equal(regulated.mu[:, :, :2], [[[10.0, 20.0], [1.0, 2.0]]])


@pytest.mark.parametrize("scale", [0, -1])
def test_regulate_encoder_rejects_invalid_speed(scale):
    with pytest.raises(ValueError, match="length_scale"):
        regulate_encoder(
            np.ones((1, 80, 1), dtype=np.float32),
            np.zeros((1, 1, 1), dtype=np.float32),
            np.ones((1, 1, 1), dtype=np.float32),
            scale,
        )
