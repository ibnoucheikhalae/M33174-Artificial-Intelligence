import numpy as np
import pytest
 
import fitness as f
 
 
# --- Helpers -----------------------------------------------------------------
def static_gait(pose=None):
    """Every frame is the same pose (default: all zeros)."""
    pose = np.zeros(f.POSE_SHAPE) if pose is None else pose
    return np.tile(pose, (f.NUM_FRAMES, 1, 1))
 
 
def interpolated_gait(targets):
    """Piecewise-linear gait that hits the targets at frames 0, 150, 299."""
    t1 = np.linspace(0, 1, 151)[:, None, None]       # frames 0..150
    t2 = np.linspace(0, 1, 150)[1:, None, None]      # frames 151..299
    first = targets[0] + t1 * (targets[1] - targets[0])
    second = targets[1] + t2 * (targets[2] - targets[1])
    return np.concatenate([first, second])
 
 
def mirrored_gait(rng):
    """Random gait whose right legs are the exact mirror of the left legs."""
    gait = static_gait()
    left = rng.uniform(-0.5, 0.5, size=(f.NUM_FRAMES, 4, f.NUM_LEG_SEGS))
    gait[:, f.LEFT_LEGS, :] = left
    # right = (left - offsets) / signs ; signs are +/-1 so dividing == multiplying
    gait[:, f.RIGHT_MIRRORED, :] = (left - f.MIRROR_OFFSETS) * np.asarray(f.MIRROR_SIGNS)
    return gait
 
 
@pytest.fixture
def rng():
    return np.random.default_rng(1234)
 
 
# --- Config sanity -------------------------------------------------------------
def test_stub_targets_are_within_joint_limits():
    f._validate_angles(f.STUB_TARGETS, (3, *f.POSE_SHAPE))
 
 
def test_stub_middle_pose_is_mirror_symmetric():
    left = f.MIDPOINT[f.LEFT_LEGS]
    right = f.MIDPOINT[f.RIGHT_MIRRORED]
    np.testing.assert_allclose(left, right * np.asarray(f.MIRROR_SIGNS))
 
 
def test_midpoint_vector_is_flattened_midpoint():
    assert f.MIDPOINT_VECTOR.shape == (24,)
    np.testing.assert_array_equal(f.MIDPOINT_VECTOR.reshape(f.POSE_SHAPE), f.MIDPOINT)
 
 
# --- _validate_angles ----------------------------------------------------------
class TestValidateAngles:
    def test_valid_gait_passes(self):
        f._validate_angles(static_gait(), f.GAIT_SHAPE)
 
    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="Expected shape"):
            f._validate_angles(np.zeros((10, 8, 3)), f.GAIT_SHAPE)
 
    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_non_finite_raises(self, bad):
        gait = static_gait()
        gait[5, 2, 1] = bad
        with pytest.raises(ValueError, match="finite"):
            f._validate_angles(gait, f.GAIT_SHAPE)
 
    def test_above_max_raises(self):
        gait = static_gait()
        gait[0, 0, 0] = f.MAX_JOINT_ANGLES[0, 0] + 0.01
        with pytest.raises(ValueError, match="joint limits"):
            f._validate_angles(gait, f.GAIT_SHAPE)
 
    def test_below_min_raises(self):
        gait = static_gait()
        gait[0, 7, 2] = f.MIN_JOINT_ANGLES[7, 2] - 0.01
        with pytest.raises(ValueError, match="joint limits"):
            f._validate_angles(gait, f.GAIT_SHAPE)
 
    def test_limits_are_inclusive(self):
        gait = static_gait()
        gait[0] = f.MAX_JOINT_ANGLES
        gait[1] = f.MIN_JOINT_ANGLES
        f._validate_angles(gait, f.GAIT_SHAPE)
 
    def test_left_and_right_knee_limits_differ(self):
        # A knee angle of +1.5 is legal on the left but not on the right.
        gait = static_gait()
        gait[0, 0, 2] = 1.5   # L1 knee
        f._validate_angles(gait, f.GAIT_SHAPE)
        gait[0, 0, 2] = 0.0
        gait[0, 7, 2] = 1.5   # R1 knee
        with pytest.raises(ValueError):
            f._validate_angles(gait, f.GAIT_SHAPE)
 
 
# --- target_pose_penalty ------------------------------------------------------
class TestTargetPosePenalty:
    def test_exact_match_is_zero(self):
        gait = interpolated_gait(f.STUB_TARGETS)
        assert f.target_pose_penalty(gait, f.STUB_TARGETS) == pytest.approx(0.0, abs=1e-12)
 
    def test_mismatch_is_positive(self):
        gait = static_gait()  # frame 150 misses the midpoint
        assert f.target_pose_penalty(gait, f.STUB_TARGETS) > 0.0
 
    def test_ignoring_midpoint_is_clearly_penalised(self):
        # Guards the scale: a static gait must not look "almost perfect".
        assert f.target_pose_penalty(static_gait(), f.STUB_TARGETS) > 0.1
 
    def test_bigger_error_gives_bigger_penalty(self):
        targets = f.STUB_TARGETS
        near = interpolated_gait(targets)
        far = near.copy()
        near[150, 0, 0] += 0.1
        far[150, 0, 0] += 0.5
        assert f.target_pose_penalty(far, targets) > f.target_pose_penalty(near, targets)
 
    def test_only_keyframes_matter(self):
        targets = f.STUB_TARGETS
        gait = interpolated_gait(targets)
        changed = gait.copy()
        changed[10] += 0.3   # frame 10 is not a keyframe
        assert f.target_pose_penalty(changed, targets) == f.target_pose_penalty(gait, targets)
 
    @pytest.mark.parametrize("frame", [0, 150, 299])
    def test_each_keyframe_is_scored(self, frame):
        targets = f.STUB_TARGETS
        gait = interpolated_gait(targets)
        base = f.target_pose_penalty(gait, targets)
        gait[frame, 3, 1] += 0.2
        assert f.target_pose_penalty(gait, targets) > base
 
    def test_exact_value(self):
        # One joint off by 1.0 rad at one keyframe:
        # RMSE = (1 / range) / sqrt(3 keyframes * 24 joints)
        targets = f.STUB_TARGETS.copy()
        gait = interpolated_gait(targets)
        gait[0, 0, 0] = targets[0, 0, 0] + 1.0
        expected = (1.0 / f.ANGLE_RANGES[0, 0]) / np.sqrt(3 * 24)
        assert f.target_pose_penalty(gait, targets) == pytest.approx(expected)
 
    def test_returns_python_float(self):
        assert isinstance(f.target_pose_penalty(static_gait(), f.STUB_TARGETS), float)
 
    def test_bad_gait_shape_raises(self):
        with pytest.raises(ValueError):
            f.target_pose_penalty(np.zeros((10, 8, 3)), f.STUB_TARGETS)
 
    def test_bad_targets_shape_raises(self):
        with pytest.raises(ValueError):
            f.target_pose_penalty(static_gait(), np.zeros((2, 8, 3)))
 
    def test_out_of_limit_targets_raise(self):
        targets = f.STUB_TARGETS.copy()
        targets[1, 0, 0] = 10.0
        with pytest.raises(ValueError):
            f.target_pose_penalty(static_gait(), targets)
 
 
# --- smoothness_penalty --------------------------------------------------------
class TestSmoothnessPenalty:
    def test_static_gait_is_zero(self):
        assert f.smoothness_penalty(static_gait()) == 0.0
 
    def test_constant_speed_is_zero(self):
        ramp = np.linspace(-0.5, 0.5, f.NUM_FRAMES)[:, None, None]
        gait = np.broadcast_to(ramp, f.GAIT_SHAPE).copy()
        assert f.smoothness_penalty(gait) == pytest.approx(0.0, abs=1e-12)
 
    def test_joints_with_different_constant_speeds_do_not_interact(self):
        gait = static_gait()
        gait[:, 0, 0] = np.linspace(0, 0.5, f.NUM_FRAMES)
        gait[:, 1, 1] = np.linspace(0, -1.0, f.NUM_FRAMES)
        assert f.smoothness_penalty(gait) == pytest.approx(0.0, abs=1e-12)
 
    def test_jittery_gait_scores_higher_than_smooth(self, rng):
        smooth = static_gait()
        jittery = np.clip(
            rng.normal(0, 0.1, size=f.GAIT_SHAPE),
            f.MIN_JOINT_ANGLES, f.MAX_JOINT_ANGLES,
        )
        assert f.smoothness_penalty(jittery) > f.smoothness_penalty(smooth)
 
    def test_is_non_negative(self, rng):
        gait = rng.uniform(f.MIN_JOINT_ANGLES, f.MAX_JOINT_ANGLES, size=f.GAIT_SHAPE)
        assert f.smoothness_penalty(gait) >= 0.0
 
    def test_exact_value_single_joint(self):
        # One joint alternates +d, -d, +d... so its change-std is exactly d.
        d = 0.2
        gait = static_gait()
        gait[:, 0, 0] = np.where(np.arange(f.NUM_FRAMES) % 2 == 0, 0.0, d)
        # changes alternate +d / -d: mean ~ 0, std ~ d (299 values, tiny bias)
        changes = np.diff(gait[:, 0, 0])
        expected = np.std(changes) / 24
        assert f.smoothness_penalty(gait) == pytest.approx(expected)
 
    def test_bad_shape_raises(self):
        with pytest.raises(ValueError):
            f.smoothness_penalty(np.zeros((10, 8, 3)))
 
 
# --- jerk_penalty ----------------------------------------------------------------
class TestJerkPenalty:
    def test_static_gait_is_zero(self):
        assert f.jerk_penalty(static_gait()) == 0.0
 
    def test_constant_speed_is_zero(self):
        ramp = np.linspace(-0.5, 0.5, f.NUM_FRAMES)[:, None, None]
        gait = np.broadcast_to(ramp, f.GAIT_SHAPE).copy()
        assert f.jerk_penalty(gait) == pytest.approx(0.0, abs=1e-12)
 
    def test_accelerating_gait_scores_higher_than_constant_speed(self):
        t = np.linspace(0, 1, f.NUM_FRAMES)
        ramp = static_gait()
        ramp[:, 0, 0] = t - 0.5
        curve = static_gait()
        curve[:, 0, 0] = t**2 - 0.5
        assert f.jerk_penalty(curve) > f.jerk_penalty(ramp)
 
    def test_sudden_spike_is_penalised(self):
        gait = static_gait()
        gait[100, 0, 0] = 0.8
        assert f.jerk_penalty(gait) > 0.0
 
    def test_bad_shape_raises(self):
        with pytest.raises(ValueError):
            f.jerk_penalty(np.zeros((10, 8, 3)))
 
 
# --- symmetry_penalty ------------------------------------------------------------
class TestSymmetryPenalty:
    def test_static_zero_gait_is_symmetric(self):
        assert f.symmetry_penalty(static_gait()) == 0.0
 
    def test_stub_middle_pose_is_symmetric(self):
        assert f.symmetry_penalty(static_gait(f.MIDPOINT)) == pytest.approx(0.0)
 
    def test_random_mirrored_gait_is_symmetric(self, rng):
        gait = mirrored_gait(rng)
        f._validate_angles(gait, f.GAIT_SHAPE)  # make sure the helper stays legal
        assert f.symmetry_penalty(gait) == pytest.approx(0.0, abs=1e-12)
 
    def test_breaking_symmetry_increases_penalty(self, rng):
        gait = mirrored_gait(rng)
        gait[:, 4, 0] += 0.3  # R4 hip yaw
        assert f.symmetry_penalty(gait) > 0.0
 
    def test_exact_value_for_one_perturbed_joint(self, rng):
        # R1 (index 7) hip yaw off by delta at every frame. Mirror error is
        # delta / range, in 1 of the 12 left/right joint slots per frame.
        gait = mirrored_gait(rng)
        delta = 0.2
        gait[:, 7, 0] += delta
        expected = (delta / f.ANGLE_RANGES[0, 0]) ** 2 / 12
        assert f.symmetry_penalty(gait) == pytest.approx(expected)
 
    @pytest.mark.parametrize("right_idx, left_idx", [(7, 0), (6, 1), (5, 2), (4, 3)])
    def test_leg_pairing(self, right_idx, left_idx):
        # Perturbing a right leg must be matched against the correct left leg:
        # nudging the left partner by the mirrored amount restores symmetry.
        gait = static_gait()
        gait[:, right_idx, 0] = 0.4
        assert f.symmetry_penalty(gait) > 0.0
        gait[:, left_idx, 0] = 0.4 * f.MIRROR_SIGNS[0]
        assert f.symmetry_penalty(gait) == pytest.approx(0.0, abs=1e-12)
 
    def test_wrong_pairing_is_not_symmetric(self):
        # L1 mirrored by R4 (instead of R1) should NOT count as symmetric.
        gait = static_gait()
        gait[:, 0, 0] = 0.4 * f.MIRROR_SIGNS[0]   # L1
        gait[:, 4, 0] = 0.4                       # R4
        assert f.symmetry_penalty(gait) > 0.0
 
    def test_custom_mirror_signs(self):
        # Identical angles on both sides are symmetric only if signs are all +1.
        gait = static_gait()
        gait[:, f.LEFT_LEGS, 0] = 0.3
        gait[:, f.RIGHT_MIRRORED, 0] = 0.3
        assert f.symmetry_penalty(gait) > 0.0
        assert f.symmetry_penalty(gait, mirror_signs=(1.0, 1.0, 1.0)) == pytest.approx(0.0)
 
    def test_custom_mirror_offsets(self):
        gait = static_gait()
        gait[:, f.LEFT_LEGS, 1] = 0.2   # left pitch = right pitch (0) + 0.2
        assert f.symmetry_penalty(gait) > 0.0
        assert f.symmetry_penalty(gait, mirror_offsets=(0.0, 0.2, 0.0)) == pytest.approx(0.0)
 
    def test_is_non_negative(self, rng):
        gait = rng.uniform(f.MIN_JOINT_ANGLES, f.MAX_JOINT_ANGLES, size=f.GAIT_SHAPE)
        assert f.symmetry_penalty(gait) >= 0.0
 
    def test_random_gait_less_symmetric_than_mirrored(self, rng):
        random_gait = rng.uniform(f.MIN_JOINT_ANGLES, f.MAX_JOINT_ANGLES, size=f.GAIT_SHAPE)
        assert f.symmetry_penalty(random_gait) > f.symmetry_penalty(mirrored_gait(rng))
 
    @pytest.mark.parametrize("signs", [(1.0, 1.0), (1.0, 1.0, 1.0, 1.0)])
    def test_bad_mirror_signs_length_raises(self, signs):
        with pytest.raises(ValueError, match="mirror_signs"):
            f.symmetry_penalty(static_gait(), mirror_signs=signs)
 
    def test_bad_mirror_offsets_length_raises(self):
        with pytest.raises(ValueError, match="mirror_offsets"):
            f.symmetry_penalty(static_gait(), mirror_offsets=(0.0, 0.0))
 
    def test_bad_shape_raises(self):
        with pytest.raises(ValueError):
            f.symmetry_penalty(np.zeros((10, 8, 3)))
 
 
# --- fitness_function ----------------------------------------------------------------
class TestFitnessFunction:
    def test_returns_python_float(self):
        assert isinstance(f.fitness_function(static_gait()), float)
 
    def test_equals_base_minus_weighted_penalties(self, rng):
        gait = rng.uniform(f.MIN_JOINT_ANGLES, f.MAX_JOINT_ANGLES, size=f.GAIT_SHAPE)
        expected = (
            f.BASE_FITNESS
            - f.TARGET_WEIGHT * f.target_pose_penalty(gait, f.STUB_TARGETS)
            - f.SMOOTHNESS_WEIGHT * f.smoothness_penalty(gait)
            - f.SYMMETRY_WEIGHT * f.symmetry_penalty(gait)
            - f.JERK_WEIGHT * f.jerk_penalty(gait)
        )
        assert f.fitness_function(gait) == pytest.approx(expected)
 
    def test_weights_are_respected(self, monkeypatch, rng):
        gait = rng.uniform(f.MIN_JOINT_ANGLES, f.MAX_JOINT_ANGLES, size=f.GAIT_SHAPE)
        for name in ("TARGET_WEIGHT", "SMOOTHNESS_WEIGHT", "SYMMETRY_WEIGHT", "JERK_WEIGHT"):
            monkeypatch.setattr(f, name, 0.0)
        assert f.fitness_function(gait) == f.BASE_FITNESS
 
        monkeypatch.setattr(f, "JERK_WEIGHT", 1.0)
        assert f.fitness_function(gait) == pytest.approx(f.BASE_FITNESS - f.jerk_penalty(gait))
 
    def test_perfect_gait_is_capped_by_base_fitness(self):
        # Static zero gait vs all-zero targets: no penalties at all.
        zero_targets = np.zeros((3, *f.POSE_SHAPE))
        assert f.fitness_function(static_gait(), zero_targets) == pytest.approx(f.BASE_FITNESS)
 
    def test_ideal_gait_scores_near_base_fitness(self):
        # Hits all targets and is symmetric -> only the small smoothness cost remains.
        gait = interpolated_gait(f.STUB_TARGETS)
        score = f.fitness_function(gait)
        expected = f.BASE_FITNESS - f.SMOOTHNESS_WEIGHT * f.smoothness_penalty(gait)
        assert score == pytest.approx(expected)
        assert score > 0.9 * f.BASE_FITNESS
 
    def test_good_gait_beats_random_gait(self, rng):
        good = interpolated_gait(f.STUB_TARGETS)
        random_gait = rng.uniform(f.MIN_JOINT_ANGLES, f.MAX_JOINT_ANGLES, size=f.GAIT_SHAPE)
        assert f.fitness_function(good) > f.fitness_function(random_gait)
 
    def test_good_gait_beats_static_gait(self):
        good = interpolated_gait(f.STUB_TARGETS)
        assert f.fitness_function(good) > f.fitness_function(static_gait())
 
    def test_asymmetry_lowers_fitness(self):
        good = interpolated_gait(f.STUB_TARGETS)
        skewed = good.copy()
        skewed[20:40, 7, 0] += 0.4   # wobble R1 mid-gait, away from keyframes
        assert f.fitness_function(skewed) < f.fitness_function(good)
 
    def test_does_not_mutate_inputs(self, rng):
        gait = rng.uniform(f.MIN_JOINT_ANGLES, f.MAX_JOINT_ANGLES, size=f.GAIT_SHAPE)
        gait_copy, targets_copy = gait.copy(), f.STUB_TARGETS.copy()
        f.fitness_function(gait)
        np.testing.assert_array_equal(gait, gait_copy)
        np.testing.assert_array_equal(f.STUB_TARGETS, targets_copy)
 
    def test_is_deterministic(self, rng):
        gait = rng.uniform(f.MIN_JOINT_ANGLES, f.MAX_JOINT_ANGLES, size=f.GAIT_SHAPE)
        assert f.fitness_function(gait) == f.fitness_function(gait)
 
    def test_invalid_gait_raises(self):
        gait = static_gait()
        gait[0, 0, 0] = np.nan
        with pytest.raises(ValueError):
            f.fitness_function(gait)
 