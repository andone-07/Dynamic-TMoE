import torch
from torch import nn
import torch.nn.functional as F

class MMDDriftDetector(nn.Module):
    """
    Drift detector based on MMD (Maximum Mean Discrepancy)
    """
    def __init__(self, d_model, window_size=576, 
                 history_size=50, k_sigma=3.0):
        """
        Args:
            d_model: feature dimension
            window_size: window size for collecting samples and detecting drift
            history_size: size of the historical MMD score buffer
            k_sigma: k value in the k-Sigma rule, threshold = mean + k * std
        """
        super().__init__()
        self.d_model = d_model
        self.window_size = window_size
        self.history_size = history_size
        self.k_sigma = k_sigma
        
        self.register_buffer('locked_bandwidth', torch.tensor(0.0))
        self.register_buffer('bandwidth_is_locked', torch.tensor(False))

        self.register_buffer('mmd_history', torch.zeros(history_size))
        self.register_buffer('history_filled', torch.tensor(0))

        self.reference_window = None
        self.register_buffer('reference_filled', torch.tensor(0))
        self.register_buffer('is_calibrated', torch.tensor(False))

        self.current_window = None
        self.register_buffer('current_filled', torch.tensor(0))

        self.register_buffer('total_samples_seen', torch.tensor(0))
        self.register_buffer('last_mmd_score', torch.tensor(0.0))
        self.register_buffer('drift_count', torch.tensor(0))

        self.current_sample_indices = []
        self.register_buffer('num_patches_per_sample', torch.tensor(0))
    
    def add_reference_samples(self, samples):
        """
        Args:
            samples: [N, P, D] - sampled patch embeddings
        """
        N, P, D = samples.shape
        
        if self.num_patches_per_sample == 0:
            self.num_patches_per_sample.fill_(P)
        
        if self.reference_window is None:
            self.reference_window = torch.zeros(
                self.window_size, P, D, 
                device=samples.device, dtype=samples.dtype
            )
            self.register_buffer('_reference_window', self.reference_window, persistent=False)

        samples = samples.detach().to(self.reference_window.device)

        remaining = self.window_size - self.reference_filled
        if remaining > 0:
            to_add = min(N, remaining)
            self.reference_window[self.reference_filled:self.reference_filled + to_add] = samples[:to_add]
            self.reference_filled += to_add

            if self.reference_filled >= self.window_size:
                self.is_calibrated.fill_(True)

                ref_mean = self.reference_window.mean(dim=2, keepdim=True)
                ref_std = self.reference_window.std(dim=2, keepdim=True)
                ref_reduced = torch.cat([ref_mean, ref_std], dim=2)
                ref_data = ref_reduced.flatten(1, 2)

                median_dist = self._compute_median_pairwise_distance_single(ref_data)
                bandwidth_value = max(median_dist, 1e-6)
                self.locked_bandwidth.fill_(bandwidth_value)
                self.bandwidth_is_locked.fill_(True)
    
    def add_current_samples_batch(self, samples, start_patch_indices=None):
        """
        Args:
            samples: [N, P, D] - multiple patch embedding sequences
            start_patch_indices: [N] - global indices of the first patch for each sample
        """
        if not self.is_calibrated:
            self.add_reference_samples(samples)
            return

        if self.current_window is None:
            N, P, D = samples.shape
            self.current_window = torch.zeros(
                self.window_size, P, D, 
                device=samples.device, dtype=samples.dtype
            )
            self.register_buffer('_current_window', self.current_window, persistent=False)
        
        N, P, D = samples.shape
        samples = samples.detach().to(self.current_window.device)

        if self.num_patches_per_sample == 0:
            self.num_patches_per_sample.fill_(P)
        
        for i in range(N):
            if self.current_filled < self.window_size:
                self.current_window[self.current_filled] = samples[i]
                self.current_sample_indices.append(start_patch_indices[i].item())
                self.current_filled += 1
            
            self.total_samples_seen += 1

    def _compute_median_pairwise_distance(self, X, Y):
        """
        Compute the median pairwise distance between two sets of samples
        Args:
            X: [n, d] - first set of samples
            Y: [m, d] - second set of samples    
        Returns:
            median_dist: float - median distance
        """
        n, m = X.shape[0], Y.shape[0]
        if n > 100 or m > 100:
            n_sample = min(n, 100)
            m_sample = min(m, 100)
            indices_x = torch.randperm(n)[:n_sample]
            indices_y = torch.randperm(m)[:m_sample]
            X_sample = X[indices_x]
            Y_sample = Y[indices_y]
        else:
            X_sample = X
            Y_sample = Y
        
        XX = (X_sample ** 2).sum(dim=1, keepdim=True)
        YY = (Y_sample ** 2).sum(dim=1, keepdim=True)
        XY = torch.mm(X_sample, Y_sample.t())
        pairwise_sq_dists = XX + YY.t() - 2 * XY
        pairwise_sq_dists = torch.clamp(pairwise_sq_dists, min=0.0)
        median_dist = torch.sqrt(torch.median(pairwise_sq_dists))
        
        return median_dist.item()
    
    def _compute_median_pairwise_distance_single(self, X):
        """
        Compute the median pairwise distance within a single set of samples
        Args:
            X: [n, d] - samples
        Returns:
            median_dist: float - median distance
        """
        n = X.shape[0]

        if n > 100:
            n_sample = min(n, 100)
            indices = torch.randperm(n)[:n_sample]
            X_sample = X[indices]
        else:
            X_sample = X

        pairwise_sq_dists = torch.cdist(X_sample, X_sample, p=2) ** 2
        triu_indices = torch.triu_indices(X_sample.shape[0], X_sample.shape[0], offset=1)
        pairwise_sq_dists_triu = pairwise_sq_dists[triu_indices[0], triu_indices[1]]
        pairwise_sq_dists_triu = torch.clamp(pairwise_sq_dists_triu, min=0.0)
        median_dist = torch.sqrt(torch.median(pairwise_sq_dists_triu))
        
        return median_dist.item()
    
    def _compute_mmd_squared(self, X, Y):
        """
        Compute MMD² (Maximum Mean Discrepancy squared)
        Args:
            X: [n, d] - reference distribution samples
            Y: [m, d] - current distribution samples
        Returns:
            mmd_squared: float - MMD² statistic
        """
        n = X.shape[0]
        m = Y.shape[0]

        if self.bandwidth_is_locked:
            bandwidth = self.locked_bandwidth.item()
        else:
            median_dist = self._compute_median_pairwise_distance_single(X)
            bandwidth = max(median_dist, 1e-6)

        XX_dists = torch.cdist(X, X, p=2) ** 2  # [n, n]
        K_XX = torch.exp(-XX_dists / (2 * bandwidth ** 2))
        term1 = (K_XX.sum() - K_XX.diag().sum()) / (n * (n - 1))

        YY_dists = torch.cdist(Y, Y, p=2) ** 2  # [m, m]
        K_YY = torch.exp(-YY_dists / (2 * bandwidth ** 2))
        term2 = (K_YY.sum() - K_YY.diag().sum()) / (m * (m - 1))

        XY_dists = torch.cdist(X, Y, p=2) ** 2  # [n, m]
        K_XY = torch.exp(-XY_dists / (2 * bandwidth ** 2))
        term3 = K_XY.mean()
        
        mmd_squared = term1 + term2 - 2 * term3
        mmd_squared = max(mmd_squared.item(), 0.0)
        
        return mmd_squared
    
    def _add_mmd_to_history(self, mmd_score):
        """
        Add MMD score to history buffer
        """
        if self.history_filled < self.history_size:
            self.mmd_history[self.history_filled] = mmd_score
            self.history_filled += 1
        else:
            self.mmd_history = torch.roll(self.mmd_history, -1, dims=0)
            self.mmd_history[-1] = mmd_score
    
    def _compute_dynamic_threshold(self):
        """
        Compute dynamic threshold using k-Sigma rule
        """
        min_samples = max(10, int(self.history_size * 0.2))
        
        if self.history_filled < min_samples:
            if self.history_filled > 0: 
                return self.mmd_history[:self.history_filled].mean().item()
            else:
                return 0.1

        valid_history = self.mmd_history[:self.history_filled]

        mean_mmd = valid_history.mean()
        std_mmd = valid_history.std()

        threshold = mean_mmd + self.k_sigma * std_mmd
        threshold = max(threshold.item(), 1e-6)
        
        return threshold
    
    def should_check_drift(self):
        return (self.is_calibrated and 
                self.current_filled >= self.window_size)
    
    def check_drift(self):
        """
        Check distribution drift
        Returns:
            drift_detected: bool - whether drift is detected
            mmd_score: float - MMD² score
            drift_patch_indices: list[int] - list of global indices of patches involved in drift
        """
        if not self.should_check_drift():
            return None

        sample_size = max(1, int(self.window_size * 0.1))
        sample_indices = torch.randperm(self.window_size, device=self.reference_window.device)[:sample_size]

        ref_sampled = self.reference_window[sample_indices]
        cur_sampled = self.current_window[sample_indices]

        ref_mean = ref_sampled.mean(dim=2, keepdim=True)
        ref_std = ref_sampled.std(dim=2, keepdim=True)
        ref_reduced = torch.cat([ref_mean, ref_std], dim=2)
        ref_data = ref_reduced.flatten(1, 2)
        
        cur_mean = cur_sampled.mean(dim=2, keepdim=True)
        cur_std = cur_sampled.std(dim=2, keepdim=True)
        cur_reduced = torch.cat([cur_mean, cur_std], dim=2)
        cur_data = cur_reduced.flatten(1, 2)
        
        mmd_squared = self._compute_mmd_squared(ref_data, cur_data)

        self.last_mmd_score.fill_(mmd_squared)
        self._add_mmd_to_history(mmd_squared)

        dynamic_threshold = self._compute_dynamic_threshold()

        drift_detected = mmd_squared > dynamic_threshold
        
        if drift_detected:
            self.drift_count += 1

        drift_patch_indices = []
        if drift_detected and len(self.current_sample_indices) == self.window_size:
            num_patches = self.num_patches_per_sample.item()
            for sample_idx, start_patch_idx in enumerate(self.current_sample_indices):
                for p in range(num_patches):
                    drift_patch_indices.append(start_patch_idx + p)

        return drift_detected, mmd_squared, drift_patch_indices
    
    def slide_window(self):
        self.reference_window.copy_(self.current_window)

        self.current_filled.fill_(0)

        self.current_sample_indices.clear()

        ref_mean = self.reference_window.mean(dim=2, keepdim=True)
        ref_std = self.reference_window.std(dim=2, keepdim=True)
        ref_reduced = torch.cat([ref_mean, ref_std], dim=2)
        ref_data = ref_reduced.flatten(1, 2)

        median_dist = self._compute_median_pairwise_distance_single(ref_data)
        bandwidth_value = max(median_dist, 1e-6)
        self.locked_bandwidth.fill_(bandwidth_value)
        self.bandwidth_is_locked.fill_(True)
    
    def get_drift_data(self):
        """
        Returns:
            ref_data: [window_size, P, D]
            cur_data: [window_size, P, D]
        """
        if not self.is_calibrated or self.current_filled < self.window_size:
            return None, None
        
        return self.reference_window.clone(), self.current_window.clone()


class DriftPatternProfiler(nn.Module):
    """
    Drift Pattern Profiler: Analyzes the drift patterns to determine the most suitable expert type
    """
    def __init__(self):
        super().__init__()
    
    def analyze_pattern(self, model_output, target_data):
        """
        Args:
            model_output: [N, P, D]
            target_data: [N, P, D]     
        Returns:
            expert_type: str
            analysis_scores: dict
        """
        diff = target_data - model_output
        batch_size, seq_len, dim = diff.shape
        diff_flat = diff.detach().permute(0, 2, 1).reshape(-1, seq_len)

        diff_mean = diff_flat.mean(dim=1, keepdim=True)
        diff_std = diff_flat.std(dim=1, keepdim=True) + 1e-6
        diff_norm = (diff_flat - diff_mean) / diff_std

        t = torch.linspace(-1, 1, seq_len, device=diff.device).unsqueeze(0)
        t_var = torch.sum(t ** 2)
        slope = torch.sum(t * diff_norm, dim=1, keepdim=True) / t_var
        trend_fit = slope * t

        ss_total = torch.sum(diff_norm ** 2, dim=1)
        ss_res_trend = torch.sum((diff_norm - trend_fit) ** 2, dim=1)
        trend_score = torch.clamp(1 - (ss_res_trend / ss_total), 0, 1).mean().item()

        fft_out = torch.fft.rfft(diff_norm, dim=1)
        power_spectrum = torch.abs(fft_out) ** 2

        ac_power = power_spectrum[:, 1:]
        total_ac_energy = torch.sum(ac_power, dim=1, keepdim=True) + 1e-8

        topk_energy, _ = torch.topk(ac_power, k=min(3, ac_power.shape[1]), dim=1)
        seasonality_score = (torch.sum(topk_energy, dim=1, keepdim=True) / total_ac_energy).mean().item()

        half_idx = ac_power.shape[1] // 2
        high_freq_energy = torch.sum(ac_power[:, half_idx:], dim=1, keepdim=True)
        fluctuation_score = (high_freq_energy / total_ac_energy).mean().item()

        scores = {
            'trend': trend_score,
            'seasonality': seasonality_score,
            'fluctuation': fluctuation_score
        }

        if trend_score > 0.4:
            expert_type = 'trend'
        elif seasonality_score > 0.3:
            expert_type = 'seasonality'
        elif fluctuation_score > 0.3:
            expert_type = 'fluctuation'
        else:
            expert_type = max(scores, key=scores.get)
        
        return expert_type, scores
