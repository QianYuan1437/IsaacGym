from legged_gym.envs.g1.g1_config import G1RoughCfg, G1RoughCfgPPO


class G1SelfCollisionRoughCfg(G1RoughCfg):
    """G1 (12-dof) config with self-collision avoidance rewards.

    Reproduction of Khazoom et al. (2022) "Humanoid self-collision avoidance
    using whole-body control with control barrier functions" in the RL setting.

    Key differences vs. the stock G1RoughCfg:
      * only_positive_rewards = False
            -> negative (penalty) terms propagate a real gradient. This is what
               makes the CBF/APF self-collision penalties effective; with the
               stock `True` they would be clipped away by compute_reward().
      * self_collision.pairs
            -> collision body pairs + per-pair safety distance, mirroring the
               paper's sphere/capsule collision pairs and their radii.
      * rewards.scales.self_collision_cbf / self_collision_apf
            -> CBF (paper method) and APF (baseline, paper Eq. 23) penalties.
               Enable exactly one of the two for a clean comparison.
    """

    class rewards(G1RoughCfg.rewards):
        only_positive_rewards = False
        class scales(G1RoughCfg.rewards.scales):
            # existing contact-force based self-collision penalty (legged_gym)
            collision = -0.5
            # CBF-style exponential barrier on link-to-link distance (paper method)
            self_collision_cbf = -0.5
            # APF-style quadratic potential, active within safe_distance (baseline)
            self_collision_apf = 0.0

    class self_collision:
        # default safety distance [m] for pairs without an explicit value
        safe_distance = 0.10
        # CBF barrier steepness: penalty = exp(-barrier_alpha * (d - safe))
        barrier_alpha = 8.0
        # velocity-aware term weight: (1 + approach_beta * max(0, -d_dot))
        approach_beta = 1.0
        # clamp for the finite-difference approach rate [m/s]
        max_approach_rate = 2.0

        # collision pairs: (body_a, body_b, safety_distance_m)
        # body names are matched against the loaded URDF; pairs whose bodies were
        # merged away by collapse_fixed_joints are skipped automatically at init.
        # Leg-leg pairs are active for the 12-dof G1; arm-torso / arm-leg pairs
        # activate automatically if a URDF with actuated arms (e.g. g1_29dof) is used.
        pairs = [
            # swing-leg self-collision avoidance (paper's walking experiments)
            ('left_hip_pitch_link', 'right_hip_pitch_link', 0.12),  # thigh - thigh
            ('left_knee_link', 'right_knee_link', 0.10),            # shin - shin
            ('left_hip_pitch_link', 'right_knee_link', 0.10),       # thigh - shin
            ('right_hip_pitch_link', 'left_knee_link', 0.10),       # thigh - shin
            # arm self-collisions (auto-skipped for the 12-dof URDF)
            ('left_elbow_link', 'torso_link', 0.10),
            ('right_elbow_link', 'torso_link', 0.10),
            ('left_wrist_roll_rubber_hand', 'torso_link', 0.08),
            ('right_wrist_roll_rubber_hand', 'torso_link', 0.08),
            ('left_wrist_roll_rubber_hand', 'left_knee_link', 0.08),
            ('right_wrist_roll_rubber_hand', 'right_knee_link', 0.08),
        ]


class G1SelfCollisionRoughCfgPPO(G1RoughCfgPPO):
    class runner(G1RoughCfgPPO.runner):
        experiment_name = 'g1_sc'
        run_name = ''
        max_iterations = 3000
