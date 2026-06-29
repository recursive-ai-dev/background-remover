import pytest
import numpy as np
from PIL import Image
import math

import bg_remover

def test_srgb_to_linear():
    rgb = np.array([0.0, 0.04045, 1.0])
    lin = bg_remover._srgb_to_linear(rgb)
    assert np.isclose(lin[0], 0.0)
    assert np.isclose(lin[1], 0.04045 / 12.92)
    assert np.isclose(lin[2], 1.0)

def test_linear_to_xyz():
    linear = np.array([1.0, 1.0, 1.0])
    xyz = bg_remover._linear_to_xyz(linear)
    assert np.allclose(xyz, [95.04559, 100.00000, 108.883], atol=1e-3)

def test_xyz_to_cielab():
    # Test D65 reference white
    xyz = np.array([95.047, 100.000, 108.883])
    lab = bg_remover._xyz_to_cielab(xyz)
    assert np.allclose(lab, [100.0, 0.0, 0.0], atol=1e-3)

    # Test Black
    xyz_black = np.array([0.0, 0.0, 0.0])
    lab_black = bg_remover._xyz_to_cielab(xyz_black)
    assert np.allclose(lab_black, [0.0, 0.0, 0.0], atol=1e-3)

def test_delta_e_cie76():
    lab1 = np.array([[50.0, 2.0, 3.0]])
    lab2 = np.array([[50.0, -1.0, 7.0]])
    # diff: 0, 3, -4 => 0^2 + 3^2 + (-4)^2 = 25 => sqrt = 5
    dist = bg_remover._delta_e_cie76(lab1, lab2)
    assert np.allclose(dist, [5.0])

def test_edt_1d():
    f = np.array([np.inf, 0, np.inf, np.inf, 0])
    # Distances to nearest 0:
    # index 0 -> 1 (sq: 1)
    # index 1 -> 0 (sq: 0)
    # index 2 -> 1 (sq: 1)
    # index 3 -> 1 (sq: 1)
    # index 4 -> 0 (sq: 0)
    expected = [1, 0, 1, 1, 0]
    res = bg_remover._edt_1d(f)
    assert np.allclose(res, expected)

def test_edt_2d():
    mask = np.array([
        [True, True, True],
        [True, False, True],
        [True, True, True]
    ])
    # EDT measures distance to False
    dist = bg_remover._euclidean_distance_transform(~mask)
    expected = np.array([
        [np.sqrt(2), 1, np.sqrt(2)],
        [1,          0, 1         ],
        [np.sqrt(2), 1, np.sqrt(2)]
    ])
    assert np.allclose(dist, expected)

def test_smoothstep():
    res = bg_remover._smoothstep(0.0, 1.0, np.array([-0.5, 0.0, 0.5, 1.0, 1.5]))
    assert np.isclose(res[0], 0.0)
    assert np.isclose(res[1], 0.0)
    assert np.isclose(res[2], 0.5)
    assert np.isclose(res[3], 1.0)
    assert np.isclose(res[4], 1.0)

def test_morphological_close():
    mask = np.array([
        [False, False, False, False],
        [False, True,  False, False],
        [False, False, False, False],
        [False, False, False, False],
    ])
    ring = np.array([
        [True, True, True],
        [True, False, True],
        [True, True, True],
    ])
    closed = bg_remover._morphological_close(ring, radius=1)
    assert closed[1, 1] == True # Hole filled

def test_morphological_open():
    noise = np.array([
        [False, False, False],
        [False, True,  False],
        [False, False, False],
    ])
    opened = bg_remover._morphological_open(noise, radius=1)
    assert opened[1, 1] == False # Noise removed

def test_removal_config_validation():
    with pytest.raises(ValueError, match="Tolerance"):
        bg_remover.RemovalConfig([(255, 255, 255)], -1.0, 0, False, False, True, 0)
    with pytest.raises(ValueError, match="Tolerance"):
        bg_remover.RemovalConfig([(255, 255, 255)], 250.0, 0, False, False, True, 0)
    with pytest.raises(ValueError, match="Feather"):
        bg_remover.RemovalConfig([(255, 255, 255)], 10.0, -1, False, False, True, 0)
    with pytest.raises(ValueError, match="At least one target color"):
        bg_remover.RemovalConfig([], 10.0, 0, False, False, True, 0)

def test_crop_to_content():
    arr = np.zeros((10, 10, 4), dtype=np.uint8)
    arr[4:6, 4:6] = [255, 0, 0, 255]
    img = Image.fromarray(arr)
    cropped = bg_remover._crop_to_content(img, padding=1)
    assert cropped.size == (4, 4)

def test_remove_background_solid(tmp_path):
    arr = np.zeros((10, 10, 4), dtype=np.uint8)
    arr[:, :5] = [255, 255, 255, 255]
    arr[:, 5:] = [0, 0, 0, 255]

    img_path = str(tmp_path / "test_input.png")
    out_path = str(tmp_path / "test_output.png")
    Image.fromarray(arr).save(img_path)

    config = bg_remover.RemovalConfig(
        target_colors=[(255, 255, 255)],
        tolerance=2.0,
        feather=0,
        invert=False,
        crop=False,
        use_perceptual=False,
        morphology=0
    )

    res = bg_remover.remove_background(img_path, out_path, config)
    assert res['removed'] == 50
    assert res['total'] == 100

    out_img = Image.open(out_path)
    out_arr = np.array(out_img)
    assert np.all(out_arr[:, :5, 3] == 0)
    assert np.all(out_arr[:, 5:, 3] == 255)

def test_k_means_detection(tmp_path):
    arr = np.zeros((20, 20, 3), dtype=np.uint8)
    arr[:15, :] = [255, 0, 0]
    arr[15:, :] = [0, 0, 255]

    img_path = str(tmp_path / "test_kmeans.png")
    Image.fromarray(arr).save(img_path)

    colors = bg_remover.detect_dominant_colors(img_path, k=2, sample_rate=1)
    assert len(colors) == 2
    r, g, b = colors[0]
    assert r > 200 and g < 50 and b < 50

    r2, g2, b2 = colors[1]
    assert r2 < 50 and g2 < 50 and b2 > 200

def test_batch_process(tmp_path):
    in_dir = tmp_path / "test_in"
    out_dir = tmp_path / "test_out"
    in_dir.mkdir()

    arr = np.zeros((10, 10, 4), dtype=np.uint8)
    arr[:, :] = [255, 255, 255, 255]
    Image.fromarray(arr).save(in_dir / "1.png")
    Image.fromarray(arr).save(in_dir / "2.png")

    config = bg_remover.RemovalConfig(
        target_colors=[(255, 255, 255)],
        tolerance=5.0,
        feather=0,
        invert=False,
        crop=False,
        use_perceptual=True,
        morphology=0
    )

    results = bg_remover.batch_process_parallel(str(in_dir), str(out_dir), config, max_workers=2)
    assert len(results) == 2
    assert all(r['ok'] for r in results)

    assert (out_dir / "1.png").exists()
    assert (out_dir / "2.png").exists()
