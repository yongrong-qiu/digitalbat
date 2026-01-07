import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import spearmanr

class Generate_Synthetic_Trajecories():
    """
    Based on an anchor flight path, generate synthetic trajectories
    """
    def __init__(
        self,
        anchor_flight_path, # numpy array, shape: (T, 3)
        n_samples=1000,
        warp_strength=0.05, # 0.05
        angle_std=0.05, # 0.05
        curvature_mag=0.02, # 0.02
        seed=123,
    ):
        self.anchor_flight_path = anchor_flight_path
        self.n_samples = n_samples
        self.warp_strength = warp_strength
        self.angle_std = angle_std
        self.curvature_mag = curvature_mag
        self.seed = seed

        self.synthetic_trajs = None

    @staticmethod
    def temporal_warp(traj, warp_strength=0.05, seed=123):
        """
        Nonlinear time reparameterization.
        traj: (T, 3), columns = x, y, z
        """
        T = traj.shape[0]
        t = np.linspace(0, 1, T)

        # Smooth random warp
        rng = np.random.default_rng(seed)
        noise = rng.normal(size=T)
        # np.random.seed(seed)
        # noise = np.random.randn(T)
        noise = np.cumsum(noise)
        noise = noise / np.max(np.abs(noise))
        t_warp = t + warp_strength * noise
        # Enforce monotonicity
        t_warp = np.clip(t_warp, 0, 1)
        t_warp = np.maximum.accumulate(t_warp)
        # Interpolate
        warped = np.zeros_like(traj)
        for d in range(3):
            f = interp1d(t, traj[:, d], kind='cubic', fill_value="extrapolate")
            warped[:, d] = f(t_warp)

        return warped

    @staticmethod
    def directional_perturbation(traj, angle_std=0.05, seed=123):
        """
        Small angular perturbations to velocity vectors.
        """
        vel = np.diff(traj, axis=0)
        vel_norm = np.linalg.norm(vel, axis=1, keepdims=True)
        vel_unit = vel / (vel_norm + 1e-8)

        # Random rotation vectors
        rng = np.random.default_rng(seed)
        noise = rng.normal(size=vel_unit.shape) * angle_std
        # np.random.seed(seed)
        # noise = np.random.randn(*vel_unit.shape) * angle_std
        vel_perturbed = vel_unit + noise
        vel_perturbed /= (np.linalg.norm(vel_perturbed, axis=1, keepdims=True)+ 1e-8)
        # Reintegrate
        traj_new = [traj[0]]
        for v, speed in zip(vel_perturbed, vel_norm):
            traj_new.append(traj_new[-1] + v * speed)

        return np.stack(traj_new)

    # def curvature_perturbation(traj, magnitude=0.3, smoothness=0.1, seed=123):
    #     """
    #     Adds smooth orthogonal displacement.
    #     magnitude: usually 0.01 - 0.05
    #     smoothness: usually 0.05 - 0.10
    #     """
    #     T = traj.shape[0]

    #     path_length = np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))
    #     magnitude = magnitude * path_length / T
    #     smoothness = int(smoothness * T)

    #     # Tangent vectors
    #     tangent = np.gradient(traj, axis=0)
    #     tangent /= (np.linalg.norm(tangent, axis=1, keepdims=True) + 1e-8)
    #     # Random orthogonal directions
    #     np.random.seed(seed)
    #     rand_vec = np.random.randn(T, 3)
    #     ortho = rand_vec - np.sum(rand_vec * tangent, axis=1, keepdims=True) * tangent
    #     ortho /= np.linalg.norm(ortho, axis=1, keepdims=True)
    #     # Smooth envelope
    #     noise = np.random.randn(T)
    #     kernel = np.ones(smoothness) / smoothness
    #     noise = np.convolve(noise, kernel, mode='same')
    #     noise = noise / np.max(np.abs(noise))

    #     displacement = magnitude * noise[:, None] * ortho
    #     return traj + displacement

    @staticmethod
    def curvature_perturbation(
        traj,
        magnitude=0.02,
        smoothness=0.1,
        seed=123,
    ):
        """
        Curvature-only perturbation:
        - preserves step length (speed)
        - preserves total path length
        - changes geometry only (turning / curvature)

        magnitude: typical 0.005 – 0.05  (radians, effectively)
        smoothness: fraction of trajectory length used for smoothing
        """

        rng = np.random.default_rng(seed)
        T = traj.shape[0]

        # --- Original velocity and step lengths ---
        vel = np.diff(traj, axis=0)                 # (T-1, 3)
        step_len = np.linalg.norm(vel, axis=1)      # (T-1,)
        tangent = vel / (step_len[:, None] + 1e-8)  # unit tangent
        # --- Construct normal directions ---
        rand_vec = rng.normal(size=tangent.shape)
        ortho = rand_vec - np.sum(rand_vec * tangent, axis=1, keepdims=True) * tangent
        ortho /= np.linalg.norm(ortho, axis=1, keepdims=True) + 1e-8
        # --- Smooth curvature envelope ---
        smooth_T = max(3, int(smoothness * T))
        noise = rng.normal(size=T - 1)
        kernel = np.ones(smooth_T) / smooth_T
        noise = np.convolve(noise, kernel, mode="same")
        noise /= np.max(np.abs(noise)) + 1e-8
        # --- Rotate tangent vectors (small-angle approximation) ---
        # t' = cos(a)t + sin(a)n ≈ t + a n
        angle = magnitude * noise
        tangent_perturbed = tangent + angle[:, None] * ortho
        tangent_perturbed /= (
            np.linalg.norm(tangent_perturbed, axis=1, keepdims=True) + 1e-8
        )
        # --- Reintegrate trajectory with original step lengths ---
        new_vel = tangent_perturbed * step_len[:, None]

        new_traj = np.zeros_like(traj)
        new_traj[0] = traj[0]
        new_traj[1:] = new_traj[0] + np.cumsum(new_vel, axis=0)

        return new_traj

    def generate_synthetic_trajectories(self):
        trajectories = []
        for ii in range(self.n_samples):
            seed = self.seed + ii
            traj = self.anchor_flight_path.copy()
            rng = np.random.default_rng(seed + self.n_samples)
            if rng.uniform() < 0.7:
                traj = self.temporal_warp(traj, self.warp_strength, seed=seed)
            if rng.uniform() < 0.7:
                traj = self.directional_perturbation(traj, self.angle_std, seed=seed)
            if rng.uniform() < 0.7:
                traj = self.curvature_perturbation(traj, self.curvature_mag, seed=seed)
            trajectories.append(traj)

        self.synthetic_trajs = np.stack(trajectories)  # (N, T, 3)
        return self.synthetic_trajs

    def sanity_check(self):
        def mean_speed(traj):
            # traj: (T, 3)
            vel = np.diff(traj, axis=0)          # (T-1, 3)
            speed = np.linalg.norm(vel, axis=1)  # per-step speed
            return speed.mean()
        def speed_variability(traj):
            vel = np.diff(traj, axis=0)
            speed = np.linalg.norm(vel, axis=1)
            return speed.std() / (speed.mean() + 1e-8)
        def mean_curvature(traj):
            vel = np.gradient(traj, axis=0)
            acc = np.gradient(vel, axis=0)

            cross = np.cross(vel, acc)
            num = np.linalg.norm(cross, axis=1)
            denom = np.linalg.norm(vel, axis=1)**3 + 1e-8

            curvature = num / denom
            return curvature.mean()
        def path_length(traj):
            return np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))

        speed_metric = np.array([speed_variability(t) for t in self.synthetic_trajs])
        curvatures   = np.array([mean_curvature(t) for t in self.synthetic_trajs])
        lengths      = np.array([path_length(t) for t in self.synthetic_trajs])

        print("corr(speed_var, curvature):",
            spearmanr(speed_metric, curvatures)[0])
        print("corr(speed_var, length):",
            spearmanr(speed_metric, lengths)[0])
        # fig, ax = plt.subplots(figsize=(4,4))
        # ax.scatter(speed_metric, curvatures)
        # ax.set_ylim(-5, 100)

        return None

