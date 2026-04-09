function compute_cell_type_metrics(to_transfer_dir, session_filter, process_mua)
% Computes cell type metrics for out_Gunits / out_Munits .mat files.
%
% Adapted from /mnt/s5adam/Angelo/code/compute_cell_type_metrics.m to:
%   - Read out_Gunits_probe*_Dataset_*.mat (and out_Munits_*.mat if process_mua)
%     from a flat To_transfer directory rather than per-session SU_*.mat files.
%   - Drop all spatial metrics so channel_positions.npy is not required.
%   - Save a sidecar out_Gunits_..._with_metrics.mat (or _Munits_) next to
%     each input file in To_transfer.
%
% Metrics computed per unit:
%   1. spike_width_ms          - Half-width at half-max of negative phase
%   2. spike_width_quarter_ms  - Width at 25% of peak amplitude
%   3. mean_acg_positive       - Center-of-mass lag of positive ACG
%   4. firing_rate_drift_hz    - Drift-corrected firing rate
%   5. trough_to_peak_ms       - Time from negative trough to subsequent peak
%   6. burst_index             - Short-lag / long-lag ACG ratio
%   7. peak_amplitude          - Peak-to-peak on best channel (template units)
%   8. spectral_width_ms       - 1/peak_frequency of spike spectrum
%   9. n_phases                - Significant extrema count
%
% Usage:
%   compute_cell_type_metrics()
%   compute_cell_type_metrics([], 'Dataset_3_14543_240419')
%   compute_cell_type_metrics([], '', true)   % include MUA (out_Munits_*)
%
% Arguments:
%   to_transfer_dir - default '/mnt/s5adam/Angelo/NP_datasets/To_transfer'
%   session_filter  - (optional) substring of dataset name to restrict to
%   process_mua     - (optional) if true, also process out_Munits_*.mat (default false)

if nargin < 1 || isempty(to_transfer_dir)
    to_transfer_dir = '/mnt/s5adam/Angelo/NP_datasets/To_transfer';
end
if nargin < 2
    session_filter = '';
end
if nargin < 3 || isempty(process_mua)
    process_mua = false;
end

SAMPLE_RATE = 30000;  % Hz

% Discover input files
patterns = {'out_Gunits_probe*_Dataset_*.mat'};
labels   = {'good'};
if process_mua
    patterns{end+1} = 'out_Munits_probe*_Dataset_*.mat';
    labels{end+1}   = 'mua';
end

% Summary accumulators
all_session_dir = {};
all_bat = [];
all_date_str = {};
all_probe = {};
all_cluster_id = [];
all_ks_label = {};
all_spike_width = [];
all_spike_width_quarter = [];
all_mean_acg = [];
all_fr_drift = [];
all_trough_to_peak = [];
all_burst_index = [];
all_peak_amplitude = [];
all_spectral_width = [];
all_n_phases = [];
all_ptype = {};

n_success = 0;
n_fail = 0;

for fi = 1:length(patterns)
    pattern   = patterns{fi};
    ks_label  = labels{fi};
    file_list = dir(fullfile(to_transfer_dir, pattern));

    fprintf('Found %d %s files matching %s\n', length(file_list), ks_label, pattern);

    for f = 1:length(file_list)
        fname = file_list(f).name;
        fpath = fullfile(to_transfer_dir, fname);

        % Skip our own sidecar outputs (the dir glob also matches them)
        if ~isempty(strfind(fname, '_with_metrics'))
            continue;
        end

        % Parse: out_Gunits_probe<N>_Dataset_<n>_<bat>_<date>.mat
        tokens = regexp(fname, ...
            'out_[GM]units_probe(\d+)_Dataset_(\d+)_(\d+)_(\d+)\.mat', 'tokens');
        if isempty(tokens)
            warning('Could not parse %s, skipping', fname);
            n_fail = n_fail + 1;
            continue;
        end
        probe_num    = str2double(tokens{1}{1});
        dataset_idx  = tokens{1}{2};
        bat_id       = str2double(tokens{1}{3});
        date_str     = tokens{1}{4};
        dataset_name = sprintf('Dataset_%s_%s_%s', dataset_idx, tokens{1}{3}, date_str);
        probe_folder = sprintf('probe%d', probe_num);

        if ~isempty(session_filter) && isempty(strfind(dataset_name, session_filter))
            continue;
        end

        fprintf('\n=== %s / %s (%s) ===\n', dataset_name, probe_folder, ks_label);

        try
            s = load(fpath);
            if strcmp(ks_label, 'good')
                struct_field = 'out_Gunits';
            else
                struct_field = 'out_Munits';
            end
            if ~isfield(s, struct_field)
                warning('%s missing field %s, skipping', fname, struct_field);
                n_fail = n_fail + 1;
                continue;
            end

            unit_struct = s.(struct_field);
            unit_table  = struct2table(unit_struct, 'AsArray', true);
            n_units = height(unit_table);
            fprintf('  %d units\n', n_units);

            if n_units == 0
                n_success = n_success + 1;
                continue;
            end

            % Compute metrics
            [unit_table, metrics] = process_unit_table(unit_table, SAMPLE_RATE);

            % pType per unit (no spatial_spread input)
            ptype_list = cell(n_units, 1);
            for i = 1:n_units
                ptype_list{i} = classify_ptype_unit( ...
                    metrics.trough_to_peak_ms(i), metrics.spectral_width_ms(i), ...
                    metrics.firing_rate_drift_hz(i), metrics.n_phases(i));
            end
            unit_table.pType = ptype_list;

            % Accumulate summary
            for i = 1:n_units
                all_session_dir{end+1, 1} = dataset_name;
                all_bat(end+1, 1) = bat_id;
                all_date_str{end+1, 1} = date_str;
                all_probe{end+1, 1} = probe_folder;
                all_cluster_id(end+1, 1) = unit_table.cluster_id(i);
                all_ks_label{end+1, 1} = ks_label;
                all_spike_width(end+1, 1) = metrics.spike_width_ms(i);
                all_spike_width_quarter(end+1, 1) = metrics.spike_width_quarter_ms(i);
                all_mean_acg(end+1, 1) = metrics.mean_acg_positive(i);
                all_fr_drift(end+1, 1) = metrics.firing_rate_drift_hz(i);
                all_trough_to_peak(end+1, 1) = metrics.trough_to_peak_ms(i);
                all_burst_index(end+1, 1) = metrics.burst_index(i);
                all_peak_amplitude(end+1, 1) = metrics.peak_amplitude(i);
                all_spectral_width(end+1, 1) = metrics.spectral_width_ms(i);
                all_n_phases(end+1, 1) = metrics.n_phases(i);
                all_ptype{end+1, 1} = ptype_list{i};
            end

            % Save sidecar with same field name. Convert table back to a
            % struct array so scipy.io.loadmat sees the same layout as the
            % original out_Gunits files (digitalbat/datasets/data.py expects
            % element-wise indexing on cluster_id / spikeTimes_usec).
            out_struct.(struct_field) = table2struct(unit_table);
            sidecar_name = strrep(fname, '.mat', '_with_metrics.mat');
            sidecar_path = fullfile(to_transfer_dir, sidecar_name);
            save(sidecar_path, '-struct', 'out_struct');
            fprintf('  Saved %s\n', sidecar_path);
            n_success = n_success + 1;

        catch ME
            warning('Error processing %s: %s', fname, ME.message);
            for k = 1:length(ME.stack)
                fprintf('    %s (line %d)\n', ME.stack(k).name, ME.stack(k).line);
            end
            n_fail = n_fail + 1;
        end
    end
end

% --- pType summary ---
fprintf('\n=== pType summary ===\n');
fprintf('  pPYR=%d, pINT=%d, unclassified=%d, noise=%d\n', ...
    sum(strcmp(all_ptype, 'pPYR')), sum(strcmp(all_ptype, 'pINT')), ...
    sum(strcmp(all_ptype, 'unclassified')), sum(strcmp(all_ptype, '')));

% Save combined summary
summary.session_dir = all_session_dir;
summary.bat = all_bat;
summary.date_str = all_date_str;
summary.probe = all_probe;
summary.cluster_id = all_cluster_id;
summary.ks_label = all_ks_label;
summary.spike_width_ms = all_spike_width;
summary.spike_width_quarter_ms = all_spike_width_quarter;
summary.mean_acg_positive = all_mean_acg;
summary.firing_rate_drift_hz = all_fr_drift;
summary.trough_to_peak_ms = all_trough_to_peak;
summary.burst_index = all_burst_index;
summary.peak_amplitude = all_peak_amplitude;
summary.spectral_width_ms = all_spectral_width;
summary.n_phases = all_n_phases;
summary.pType = all_ptype;

summary_file = fullfile(to_transfer_dir, 'cell_type_metrics_all.mat');
save(summary_file, 'summary');
fprintf('\n========================================\n');
fprintf('Summary saved to %s\n', summary_file);
fprintf('Total units: %d\n', length(all_cluster_id));
fprintf('Files processed: %d success, %d failed\n', n_success, n_fail);
fprintf('========================================\n');

end


function [unit_table, metrics] = process_unit_table(unit_table, sample_rate)
% Compute non-spatial cell type metrics for each unit in the table.
% Requires unit_table to have columns: cluster_id, spikeTimes_usec, template.

n_units = height(unit_table);

metrics.spike_width_ms = nan(n_units, 1);
metrics.spike_width_quarter_ms = nan(n_units, 1);
metrics.mean_acg_positive = nan(n_units, 1);
metrics.firing_rate_drift_hz = nan(n_units, 1);
metrics.trough_to_peak_ms = nan(n_units, 1);
metrics.burst_index = nan(n_units, 1);
metrics.peak_amplitude = nan(n_units, 1);
metrics.spectral_width_ms = nan(n_units, 1);
metrics.n_phases = nan(n_units, 1);

ACG_N_BINS = 300;
metrics.acg_vector = nan(n_units, ACG_N_BINS);

for i = 1:n_units
    template = unit_table.template{i};  % (n_timepoints x n_channels)

    % Best channel: largest absolute extremum
    [min_vals, ~] = min(template, [], 1);
    [max_vals, ~] = max(template, [], 1);
    [neg_extreme, neg_ch] = min(min_vals);
    [pos_extreme, pos_ch] = max(max_vals);
    if abs(neg_extreme) >= abs(pos_extreme)
        best_ch = neg_ch;
        inverted = false;
    else
        best_ch = pos_ch;
        inverted = true;
    end
    waveform = template(:, best_ch);

    metrics.peak_amplitude(i) = max(waveform) - min(waveform);
    metrics.n_phases(i) = count_phases(waveform, 0.1);
    metrics.spectral_width_ms(i) = compute_spectral_width(waveform, sample_rate);
    metrics.spike_width_ms(i) = compute_spike_width(waveform, sample_rate, inverted, 0.5);
    metrics.spike_width_quarter_ms(i) = compute_spike_width(waveform, sample_rate, inverted, 0.25);
    metrics.trough_to_peak_ms(i) = compute_trough_to_peak(waveform, sample_rate, inverted);

    % --- Spike times for ACG and firing rate ---
    spike_times_usec = unit_table.spikeTimes_usec{i};
    spike_times_sec = double(spike_times_usec(:)) / 1e6;
    spike_times_sec = sort(spike_times_sec);

    if length(spike_times_sec) >= 2
        acg_pos = compute_acg(spike_times_sec, 0.001, 0.300);
        n_bins = length(acg_pos);

        bin_centers_ms = ((1:n_bins) - 0.5);
        total_count = sum(acg_pos);
        if total_count > 0
            metrics.mean_acg_positive(i) = sum(acg_pos .* bin_centers_ms) / total_count;
        end

        burst_bins = 4:min(5, n_bins);
        baseline_bins = 201:min(300, n_bins);
        if ~isempty(burst_bins) && ~isempty(baseline_bins)
            baseline_mean = mean(acg_pos(baseline_bins));
            if baseline_mean > 0
                metrics.burst_index(i) = mean(acg_pos(burst_bins)) / baseline_mean;
            end
        end

        if total_count > 0
            acg_norm = acg_pos / total_count;
            n_store = min(n_bins, ACG_N_BINS);
            metrics.acg_vector(i, 1:n_store) = acg_norm(1:n_store);
        end
    end

    metrics.firing_rate_drift_hz(i) = compute_drift_corrected_fr(spike_times_sec);
end

unit_table.spike_width_ms = metrics.spike_width_ms;
unit_table.spike_width_quarter_ms = metrics.spike_width_quarter_ms;
unit_table.mean_acg_positive = metrics.mean_acg_positive;
unit_table.firing_rate_drift_hz = metrics.firing_rate_drift_hz;
unit_table.trough_to_peak_ms = metrics.trough_to_peak_ms;
unit_table.burst_index = metrics.burst_index;
unit_table.peak_amplitude = metrics.peak_amplitude;
unit_table.spectral_width_ms = metrics.spectral_width_ms;
unit_table.n_phases = metrics.n_phases;

end


function spike_width = compute_spike_width(waveform, sample_rate, inverted, frac_amp)
if nargin < 4
    frac_amp = 0.5;
end

if inverted
    [peak_val, peak_idx] = max(waveform);
    thresh = peak_val * frac_amp;
    left_idx = NaN;
    for k = peak_idx-1:-1:1
        if waveform(k) <= thresh
            frac = (thresh - waveform(k)) / (waveform(k+1) - waveform(k));
            left_idx = k + frac;
            break;
        end
    end
    right_idx = NaN;
    for k = peak_idx+1:length(waveform)
        if waveform(k) <= thresh
            frac = (thresh - waveform(k-1)) / (waveform(k) - waveform(k-1));
            right_idx = (k-1) + frac;
            break;
        end
    end
else
    [trough_val, trough_idx] = min(waveform);
    thresh = trough_val * frac_amp;
    left_idx = NaN;
    for k = trough_idx-1:-1:1
        if waveform(k) >= thresh
            frac = (thresh - waveform(k)) / (waveform(k+1) - waveform(k));
            left_idx = k + frac;
            break;
        end
    end
    right_idx = NaN;
    for k = trough_idx+1:length(waveform)
        if waveform(k) >= thresh
            frac = (thresh - waveform(k-1)) / (waveform(k) - waveform(k-1));
            right_idx = (k-1) + frac;
            break;
        end
    end
end

if isnan(left_idx) || isnan(right_idx)
    spike_width = NaN;
else
    spike_width = (right_idx - left_idx) / sample_rate * 1000;
end

end


function ttp = compute_trough_to_peak(waveform, sample_rate, inverted)
if inverted
    [~, peak_idx] = max(waveform);
    first_idx = peak_idx;
    after_first = waveform(peak_idx:end);
    [trough_val, rel_idx] = min(after_first);
    second_idx = peak_idx + rel_idx - 1;
    if trough_val >= 0
        ttp = NaN;
        return;
    end
else
    [~, trough_idx] = min(waveform);
    first_idx = trough_idx;
    after_first = waveform(trough_idx:end);
    [peak_val, rel_idx] = max(after_first);
    second_idx = trough_idx + rel_idx - 1;
    if peak_val <= 0
        ttp = NaN;
        return;
    end
end

if rel_idx > 1 && second_idx < length(waveform)
    y1 = waveform(second_idx - 1);
    y2 = waveform(second_idx);
    y3 = waveform(second_idx + 1);
    denom = y1 - 2*y2 + y3;
    if denom ~= 0
        delta = 0.5 * (y1 - y3) / denom;
        second_idx_refined = second_idx + delta;
    else
        second_idx_refined = second_idx;
    end
else
    second_idx_refined = second_idx;
end

ttp = (second_idx_refined - first_idx) / sample_rate * 1000;

end


function acg_pos = compute_acg(spike_times_sec, bin_size, max_lag)
edges = 0:bin_size:max_lag;
n_bins = length(edges) - 1;
acg_pos = zeros(1, n_bins);
n = length(spike_times_sec);

for i = 1:n
    j = i + 1;
    while j <= n && (spike_times_sec(j) - spike_times_sec(i)) <= max_lag
        j = j + 1;
    end
    if j > i + 1
        diffs = spike_times_sec((i+1):(j-1)) - spike_times_sec(i);
        acg_pos = acg_pos + histcounts(diffs, edges);
    end
end

end


function fr = compute_drift_corrected_fr(spike_times_sec)
if isempty(spike_times_sec)
    fr = 0;
    return;
end

block_size = 30;
t_start = spike_times_sec(1);
t_end = spike_times_sec(end);
duration = t_end - t_start;

if duration <= 0
    fr = 0;
    return;
end

block_edges = t_start:block_size:(t_end + block_size);
counts = histcounts(spike_times_sec, block_edges);

active_mask = counts >= 1;
n_active_blocks = sum(active_mask);
total_spikes_active = sum(counts(active_mask));

if n_active_blocks == 0
    fr = 0;
else
    block_durations = diff(block_edges);
    total_time_active = sum(block_durations(active_mask));
    fr = total_spikes_active / total_time_active;
end

end


function spectral_width_ms = compute_spectral_width(waveform, sample_rate)
spectral_width_ms = NaN;

wf = waveform(:);
n_orig = length(wf);

peak_amp = max(abs(wf));
if peak_amp == 0
    return;
end
edge_amp = max(abs(wf(1)), abs(wf(end)));
if edge_amp > 0.2 * peak_amp
    n_taper = min(5, floor(n_orig / 4));
    taper = 0.5 * (1 - cos(pi * (0:n_taper-1)' / n_taper));
    wf(1:n_taper) = wf(1:n_taper) .* taper;
    wf(end-n_taper+1:end) = wf(end-n_taper+1:end) .* flipud(taper);
end

wf = wf - mean(wf);

n_fft = 1024;
wf_padded = zeros(n_fft, 1);
n_pad_left = floor((n_fft - n_orig) / 2);
wf_padded(n_pad_left + 1 : n_pad_left + n_orig) = wf;

Y = abs(fft(wf_padded));
n_half = floor(n_fft / 2) + 1;
mag = Y(1:n_half);

freq = (0:n_half-1)' * (sample_rate / n_fft);

freq_lo = 100;
freq_hi = 10000;
range_mask = freq >= freq_lo & freq <= freq_hi;
if ~any(range_mask)
    return;
end

mag_range = mag(range_mask);
freq_range = freq(range_mask);

[~, peak_idx] = max(mag_range);
peak_freq = freq_range(peak_idx);

if peak_freq > 0
    spectral_width_ms = 1000 / peak_freq;
end

end


function n = count_phases(waveform, frac_thresh)
wf = waveform(:);
p2p = max(wf) - min(wf);
if p2p == 0
    n = 0;
    return;
end
baseline = mean(wf([1:3, end-2:end]));
thresh = frac_thresh * p2p;

n = 0;
for i = 2:(length(wf) - 1)
    is_peak = wf(i) > wf(i-1) && wf(i) > wf(i+1);
    is_trough = wf(i) < wf(i-1) && wf(i) < wf(i+1);
    if (is_peak || is_trough) && abs(wf(i) - baseline) >= thresh
        n = n + 1;
    end
end

end


function ptype = classify_ptype_unit(ttp, sw, fr, n_phases)
% Classify a single unit as pPYR, pINT, unclassified, or noise ('').
% Spatial-spread noise criterion removed (no spatial metrics here).
%
% Noise: T2P > 0.6, FR > 50, n_phases > 3
% pINT:  T2P < 0.25 AND spectral_width < 0.7
% pPYR:  T2P > 0.25 AND spectral_width > 0.8 AND FR < 20

if ttp > 0.6 || fr > 50 || n_phases > 3
    ptype = '';
    return;
end

if isnan(ttp) || isnan(sw)
    ptype = 'unclassified';
    return;
end

if ttp < 0.25 && sw < 0.7
    ptype = 'pINT';
elseif ttp > 0.25 && sw > 0.8 && fr < 20
    ptype = 'pPYR';
else
    ptype = 'unclassified';
end

end
