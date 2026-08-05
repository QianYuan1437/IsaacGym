import torch

from legged_gym.envs.g1.g1_env import G1Robot


class G1SelfCollisionRobot(G1Robot):
    """G1 humanoid RL env with CBF / APF-style self-collision avoidance rewards.

    Reproduction of Khazoom et al. (2022), "Humanoid self-collision avoidance
    using whole-body control with control barrier functions" (IEEE-RAS Humanoids),
    inside the IsaacGym + RSL-RL legged_gym framework.

    The paper guarantees collision-free motions by enforcing Control Barrier
    Function (CBF) constraints on the signed distance between geometric
    primitives (spheres/capsules) inside a whole-body controller QP.
    Here we translate that safety notion into reward shaping:
      - CBF mode  : an exponential barrier on the link-to-link distance that
                    grows as the distance -> 0 and that also amplifies when the
                    links approach each other fast (mirrors the constraint
                    h_dot + alpha*h >= 0 of the paper).
      - APF mode  : a soft quadratic potential that is active only when the
                    distance drops below a threshold d0 (mirrors Eq. 23 of the
                    paper). Used as the baseline for comparison.

    Collision pairs and their safety distances are defined in the config
    (cfg.self_collision.pairs), resolved at init via rigid-body handles, and
    evaluated from the (already refreshed) rigid body state tensor.
    """

    def _init_buffers(self):
        super()._init_buffers()
        self._init_self_collision()

    def _init_self_collision(self):
        """Resolve the configured collision pairs to rigid-body indices."""
        sc = self.cfg.self_collision
        idx_a, idx_b, safe = [], [], []
        for pair in sc.pairs:
            name_a, name_b = pair[0], pair[1]
            safe_d = pair[2] if len(pair) > 2 else sc.safe_distance
            handle_a = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], name_a)
            handle_b = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], name_b)
            if handle_a < 0 or handle_b < 0:
                # body merged away by collapse_fixed_joints, skip
                print(f"[self_collision] pair ({name_a}, {name_b}) not found, skipped")
                continue
            idx_a.append(handle_a)
            idx_b.append(handle_b)
            safe.append(safe_d)

        self.sc_idx_a = torch.tensor(idx_a, dtype=torch.long, device=self.device)
        self.sc_idx_b = torch.tensor(idx_b, dtype=torch.long, device=self.device)
        self.sc_safe = torch.tensor(safe, dtype=torch.float, device=self.device)
        self.sc_dist_prev = torch.zeros(
            self.num_envs, len(idx_a), dtype=torch.float, device=self.device)
        if len(idx_a) == 0:
            print("[self_collision] WARNING: no valid collision pairs, ",
                  "self-collision rewards are disabled")

    def _sc_distances(self):
        """Center-to-center distances between each collision pair.
        Uses the rigid body state tensor (refreshed every step in
        G1Robot.update_feet_state -> _post_physics_step_callback).
        """
        pos = self.rigid_body_states_view[:, :, :3]  # (num_envs, num_bodies, 3)
        pos_a = pos[:, self.sc_idx_a, :]
        pos_b = pos[:, self.sc_idx_b, :]
        return torch.norm(pos_a - pos_b, dim=2)  # (num_envs, n_pairs)

    def _reward_self_collision_cbf(self):
        """CBF-style barrier penalty (paper's core contribution).

        penalty_pair = exp(-alpha * (d - safe)) * (1 + beta * max(0, -d_dot))
        * barrier is ~1 at d == safe and grows exponentially as d -> 0
        * the d_dot term mirrors h_dot + alpha*h >= 0: approaching bodies are
          penalized harder, receding bodies are not penalized extra
        """
        sc = self.cfg.self_collision
        d = self._sc_distances()
        d_dot = torch.clamp(
            (d - self.sc_dist_prev) / self.dt,
            -sc.max_approach_rate, sc.max_approach_rate)
        self.sc_dist_prev = d
        h = d - self.sc_safe
        barrier = torch.exp(-sc.barrier_alpha * h)
        approach = 1.0 + sc.approach_beta * torch.relu(-d_dot)
        return torch.sum(barrier * approach, dim=1)

    def _reward_self_collision_apf(self):
        """APF-style baseline penalty (paper Eq. 23, position-only).

        penalty_pair = max(0, safe - d)^2   (active only when d < d0)
        """
        d = self._sc_distances()
        pen = torch.relu(self.sc_safe - d) ** 2
        return torch.sum(pen, dim=1)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if len(env_ids) == 0:
            return
        # re-seed the previous-distance buffer so the approach-rate term of the
        # CBF reward is not corrupted by the discontinuity caused by a reset
        d = self._sc_distances()[env_ids]
        self.sc_dist_prev[env_ids] = d
