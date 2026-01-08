import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

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


class Time_Resolved_TRF():
    """
    Estimate how a neuron's firing depends on trajectory features (such as speed, curvature, turning angle, phase), under contrilled perturbations.
    This gives us statistical power for one anchor flight, not across several real flights.
    Each neuron gets its own TRF vector.

    # TODO
    In the future, if we want, we can add other trajectory features such as:
        signed curvature (left vs right turns)
        distance to landmarks
        torsion (3D curvature)
        nonlinear TRFs (splines, GAMs)
        phase × curvature interactions
        
    """
    def __init__(self, alpha=1.0, n_splits=5, n_shuffles=500, seed=0):
        """
        alpha: ridge regularization strength
        """
        self.alpha = alpha
        self.n_splits = n_splits
        self.n_shuffles = n_shuffles
        self.rng = np.random.default_rng(seed)

    # ---------- Trajectory feature extraction ----------

    def extract_features(self, traj):
        """
        traj: (T, 3)
        returns dict of feature arrays, each shape (T-2,)
        """
        vel = np.diff(traj, axis=0)              # (T-1, 3)
        speed = np.linalg.norm(vel, axis=1)      # (T-1,)

        # heading
        heading = vel / (speed[:, None] + 1e-8)  # (T-1, 3)

        # acceleration
        acc = np.diff(vel, axis=0)               # (T-2, 3)

        # curvature
        cross = np.cross(vel[:-1], acc)
        curvature = np.linalg.norm(cross, axis=1) / (
            np.linalg.norm(vel[:-1], axis=1) ** 3 + 1e-8
        )

        # turning angle
        dot = np.sum(heading[:-1] * heading[1:], axis=1)
        dot = np.clip(dot, -1.0, 1.0)
        turn_rate = np.arccos(dot)

        # phase (arc-length normalized)
        arc = np.concatenate([[0], np.cumsum(speed)])
        phase = arc[:-2] / arc[-1]

        return {
            "speed": speed[:-1],
            "curvature": curvature,
            "turn_rate": turn_rate,
            "phase": phase,
        }

    # ---------- TRF fitting ----------

    def fit_old(self, synthetic_trajs, synthetic_y_pred):
        """
        synthetic_trajs: (N, T, 3)
        synthetic_y_pred: (N, T, n_neurons)

        returns:
            trf_results: list of dicts (one dict per neuron)
        """
        N, T, _ = synthetic_trajs.shape
        n_neurons = synthetic_y_pred.shape[-1]

        # --- build feature matrix ---
        feature_list = []
        y_list = [[] for _ in range(n_neurons)]

        for i in range(N):
            feats = self.extract_features(synthetic_trajs[i])
            X_i = np.column_stack(list(feats.values()))  # (T-2, n_features)
            feature_list.append(X_i)

            for n in range(n_neurons):
                y_list[n].append(synthetic_y_pred[i, 1:-1, n])

        X = np.vstack(feature_list)  # (N*(T-2), n_features)
        feature_names = list(feats.keys())

        results = []

        for n in range(n_neurons):
            y = np.concatenate(y_list[n])

            # standardize
            Xs = StandardScaler().fit_transform(X)
            ys = (y - y.mean()) / (y.std() + 1e-8)

            # ridge regression
            model = Ridge(alpha=self.alpha)
            model.fit(Xs, ys)

            betas = dict(zip(feature_names, model.coef_))

            # rank correlations (robust sanity check)
            corrs = {
                name: spearmanr(X[:, i], y)[0]
                for i, name in enumerate(feature_names)
            }

            results.append({
                "betas": betas,
                "spearman": corrs,
                "r2": model.score(Xs, ys),
            })

        return results
    
    # ---------- Main fitting + statistics ----------
    def _fit_and_score(self, X, y):
        """
        Cross-validated R² and betas
        """
        kf = KFold(self.n_splits, shuffle=True, random_state=0)
        r2_scores = []
        betas = []

        for train_idx, test_idx in kf.split(X):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_test  = scaler.transform(X[test_idx])

            model = Ridge(alpha=self.alpha)
            model.fit(X_train, y[train_idx])

            r2_scores.append(model.score(X_test, y[test_idx]))
            betas.append(model.coef_)

        return np.mean(r2_scores), np.mean(betas, axis=0)

    def fit(self, synthetic_trajs, synthetic_y_pred):
        """
        synthetic_trajs: (N, T, 3)
        synthetic_y_pred: (N, T, n_neurons)

        returns:
            trf_results: list of dicts (one dict per neuron)
        """
        N, T, _ = synthetic_trajs.shape
        n_neurons = synthetic_y_pred.shape[-1]

        feature_blocks = []
        y_blocks = [[] for _ in range(n_neurons)]

        for i in range(N):
            feats = self.extract_features(synthetic_trajs[i])
            X_i = np.column_stack(list(feats.values())) # (T-2, n_features)
            feature_blocks.append(X_i)

            for n in range(n_neurons):
                y_blocks[n].append(synthetic_y_pred[i, 1:-1, n])

        feature_names = list(feats.keys())
        n_features = len(feature_names)

        X = np.vstack(feature_blocks) # (N*(T-2), n_features)
        block_sizes = [len(b) for b in feature_blocks]
        # print(f'X.shape: {X.shape}')

        results = []

        for n in range(n_neurons):
            y = np.concatenate(y_blocks[n])

            # scaler_X = StandardScaler().fit(X)
            # Xs = scaler_X.transform(X)
            # ys = (y - y.mean()) / (y.std() + 1e-8)
            # model = Ridge(alpha=self.alpha)
            # model.fit(Xs, ys)
            # beta_obs = model.coef_.copy()
            # r2 = model.score(Xs, ys)

            r2, beta_obs = self._fit_and_score(X, y)

            # ---------- NEW: beta shuffle null ----------
            beta_null = np.zeros((self.n_shuffles, n_features))
            # for s in range(self.n_shuffles):
            #     Xs_null = Xs.copy()
            #     for j in range(n_features):
            #         # shuffle feature j across trajectories
            #         start = 0
            #         perm = self.rng.permutation(N)
            #         shuffled_col = []
            #         for idx in perm:
            #             size = block_sizes[idx]
            #             shuffled_col.append(
            #                 Xs[start:start+size, j]
            #             )
            #             start += size
            #         Xs_null[:, j] = np.concatenate(shuffled_col)
            #     model_null = Ridge(alpha=self.alpha)
            #     model_null.fit(Xs_null, ys)
            #     beta_null[s] = model_null.coef_

            # shuffle each feature across time within each synthetic trajectory
            for i in range(self.n_shuffles):
                X_null = np.reshape(X.copy(), (N, T-2, n_features))
                for j in range(n_features):
                    self.rng.shuffle(X_null[:, :, j], axis=1)
                _, beta_null[i,] = self._fit_and_score(np.reshape(X_null, (-1, n_features)), y)

            # two-sided p-values
            pvals = {
                feature_names[j]:
                np.mean(np.abs(beta_null[:, j]) >= abs(beta_obs[j]))
                for j in range(n_features)
            }
            betas = dict(zip(feature_names, beta_obs))
            corrs = {
                name: spearmanr(X[:, i], y)[0]
                for i, name in enumerate(feature_names)
            }
            results.append({
                "betas": betas,
                "pvals": pvals,
                "spearman": corrs,
                "r2": r2,
            })

        return results


class Trajectory_Level_TRF():
    """
    Statistical test of Trajectory Receptive Fields (TRFs)
    for a single anchor trajectory using synthetic perturbations.
    
    This analysis uses trajectory-level features collapsing over time.
    Yet, TRFEstimator() uses time-resolved features within each synthetic trajectory (collapsing perturbations).
    """

    def __init__(
        self,
        synthetic_trajs,      # (N, T, 3)
        synthetic_y_pred,     # (N, T, n_neurons)
        feature_fns=None,          # dict: name -> function(traj) -> scalar
        alpha=1.0,            # ridge regularization
        n_splits=5,
        n_null=500,
        seed=0,
    ):
        self.synthetic_trajs = synthetic_trajs
        self.synthetic_y_pred = synthetic_y_pred
        if feature_fns:
            self.feature_fns = feature_fns
        else:
            feature_fns = {
                "speed": Trajectory_Level_TRF.mean_speed,
                "speed_var": Trajectory_Level_TRF.speed_variability,
                "curvature": Trajectory_Level_TRF.mean_curvature,
                "turn_rate": Trajectory_Level_TRF.mean_turn_rate,
                # "phase": phase_center,
            }
            self.feature_fns = feature_fns
        self.alpha = alpha
        self.n_splits = n_splits
        self.n_null = n_null
        self.rng = np.random.default_rng(seed)

        self.N, self.T, self.n_neurons = synthetic_y_pred.shape
        self.feature_names = list(feature_fns.keys())

        self.X = self._compute_feature_matrix()
        self.Y = self._compute_response_matrix()

    # -----------------------
    # Feature & response extraction
    # -----------------------
    def _compute_feature_matrix(self):
        """X: (N, n_features)"""
        X = np.zeros((self.N, len(self.feature_fns)))
        for i, traj in enumerate(self.synthetic_trajs):
            for j, fn in enumerate(self.feature_fns.values()):
                X[i, j] = fn(traj)
        return X

    def _compute_response_matrix(self):
        """
        Y: (N, n_neurons)
        Trajectory-level response = mean firing rate
        """
        return self.synthetic_y_pred.mean(axis=1)

    # -----------------------
    # Core regression
    # -----------------------
    def _fit_and_score(self, X, y):
        """
        Cross-validated R² and betas
        """
        kf = KFold(self.n_splits, shuffle=True, random_state=0)
        r2_scores = []
        betas = []

        for train_idx, test_idx in kf.split(X):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_test  = scaler.transform(X[test_idx])

            model = Ridge(alpha=self.alpha)
            model.fit(X_train, y[train_idx])

            r2_scores.append(model.score(X_test, y[test_idx]))
            betas.append(model.coef_)

        return np.mean(r2_scores), np.mean(betas, axis=0)

    # -----------------------
    # Null model
    # -----------------------
    def _null_distribution(self, y):
        """
        Feature-permutation null
        """
        null_betas = np.zeros((self.n_null, len(self.feature_names)))
        null_r2 = np.zeros(self.n_null)

        for i in range(self.n_null):
            X_null = self.X.copy()
            for j in range(X_null.shape[1]):
                self.rng.shuffle(X_null[:, j])

            null_r2[i], null_betas[i] = self._fit_and_score(X_null, y)

        return null_r2, null_betas
    
    # -----------------------
    # feature_fns
    # -----------------------
    @staticmethod
    def mean_speed(traj):
        v = np.diff(traj, axis=0)
        return np.linalg.norm(v, axis=1).mean()

    @staticmethod
    def speed_variability(traj):
        v = np.diff(traj, axis=0)
        s = np.linalg.norm(v, axis=1)
        return s.std() / (s.mean() + 1e-8)

    @staticmethod 
    def mean_curvature(traj):
        v = np.gradient(traj, axis=0)
        a = np.gradient(v, axis=0)
        num = np.linalg.norm(np.cross(v, a), axis=1)
        denom = np.linalg.norm(v, axis=1)**3 + 1e-8
        return np.mean(num / denom)

    @staticmethod
    def mean_turn_rate(traj):
        v = np.diff(traj, axis=0)
        v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
        angles = np.arccos(np.clip(np.sum(v[:-1] * v[1:], axis=1), -1, 1))
        return angles.mean()

    @staticmethod
    def phase_center(traj):
        # Anchor-normalized phase (constant by construction)
        return 0.5


    # -----------------------
    # Public API
    # -----------------------
    def run(self):
        """
        Returns results for all neurons
        """
        results = []

        for neuron_idx in range(self.n_neurons):
            y = self.Y[:, neuron_idx]

            r2_obs, beta_obs = self._fit_and_score(self.X, y)
            r2_null, beta_null = self._null_distribution(y)

            p_r2 = np.mean(r2_null >= r2_obs)

            p_betas = {
                name: np.mean(np.abs(beta_null[:, i]) >= abs(beta_obs[i]))
                for i, name in enumerate(self.feature_names)
            }

            results.append({
                "neuron": neuron_idx,
                "r2_obs": r2_obs,
                "r2_null_mean": r2_null.mean(),
                "p_r2": p_r2,
                "betas": dict(zip(self.feature_names, beta_obs)),
                "p_betas": p_betas,
            })

        return results