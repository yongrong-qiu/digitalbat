import numpy as np
import torch


def data_prep_for_mlp(all_input, all_label, train_num, valid_num, use_fourier=False):
    """
    Data preprocessing for MLP.

    Args:
        all_input: np.ndarray, shape (n_samples, n_features)
            Full input data. Positions assumed to be in columns 1 to 3.
        all_label: np.ndarray, shape (n_samples, n_outputs)
            Corresponding labels.
        train_num: int
            Number of training samples.
        valid_num: int
            Number of validation samples.
        use_fourier: bool, default=False
            If True, normalize positions (columns 1–3) to [0, 1] range instead of std.

    Returns:
        Tuple of torch.Tensors:
            train_input, valid_input, test_input,
            train_label, valid_label, test_label
    """

    # Shuffle the data
    np.random.seed(100)
    data_indices = np.random.permutation(all_input.shape[0])
    all_input = all_input[data_indices]
    all_label = all_label[data_indices]

    # Split data
    train_input = all_input[:train_num]
    train_label = all_label[:train_num]
    valid_input = all_input[train_num:train_num + valid_num]
    valid_label = all_label[train_num:train_num + valid_num]
    test_input = all_input[train_num + valid_num:]
    test_label = all_label[train_num + valid_num:]

    # Normalize input
    if use_fourier:
        # Normalize position dims (0:3) to [0, 1]
        pos_min = train_input[:, 0:3].min(axis=0)
        pos_max = train_input[:, 0:3].max(axis=0)
        pos_range = pos_max - pos_min + 1e-8  # avoid div by zero

        def normalize_pos(x):
            x[:, 0:3] = (x[:, 0:3] - pos_min) / pos_range
            return x

        train_input = normalize_pos(train_input)
        valid_input = normalize_pos(valid_input)
        test_input = normalize_pos(test_input)

        # Standardize other dims (0, 3:10)
        keep_dims = list(range(3, all_input.shape[1]))
        train_std = np.std(train_input[:, keep_dims], axis=0, keepdims=True)
        train_input[:, keep_dims] /= train_std
        valid_input[:, keep_dims] /= train_std
        test_input[:, keep_dims] /= train_std
    else:
        # Standardize all dimensions
        train_std = np.std(train_input, axis=0, keepdims=True)
        train_input /= train_std
        valid_input /= train_std
        test_input /= train_std

    # Convert to torch tensors
    train_input = torch.from_numpy(train_input).float()
    valid_input = torch.from_numpy(valid_input).float()
    test_input = torch.from_numpy(test_input).float()
    train_label = torch.from_numpy(train_label).float()
    valid_label = torch.from_numpy(valid_label).float()
    test_label = torch.from_numpy(test_label).float()

    return train_input, valid_input, test_input, train_label, valid_label, test_label