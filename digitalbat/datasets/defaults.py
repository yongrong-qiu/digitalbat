"""Default ``bat_data_loader`` config used across the analysis notebooks.

Centralizes the standard session list and the ``data_config`` dict so notebooks
can stay in sync without copy-pasting them.  Notebooks that need a subset of
sessions (e.g., ``investigate_similar_trajectories``) still pass an explicit
``session_names`` argument.
"""

DEFAULT_PARENT_FOLDER = '/mnt/s5/Adam/Angelo/NP_datasets/'

DEFAULT_SESSION_NAMES = [
    'Dataset_3_14543_240419',
    'Dataset_3_14543_240420',
    'Dataset_3_14543_240421',
    'Dataset_3_29959_240402',
    'Dataset_3_29959_240403',
    'Dataset_3_29959_240404',
    'Dataset_2_14445_231208',
    'Dataset_2_14445_231209',
    'Dataset_1_32622_231009',
    'Dataset_1_32622_231010',
    'Dataset_1_32622_231007',
    'Dataset_2_14611_231208',
    'Dataset_2_14611_231209',
    'Dataset_3_00000_240402',
    'Dataset_3_00000_240403',
    'Dataset_3_00000_240404',
    'Dataset_3_00000_240405',
    'Dataset_3_29959_240405',
]


def make_data_config(session_names=None, parent_folder=DEFAULT_PARENT_FOLDER,
                     max_acceleration=40):
    """Return the standard ``data_config`` dict for ``bat_data_loader``."""
    if session_names is None:
        session_names = list(DEFAULT_SESSION_NAMES)
    return dict(
        use_speed_head_dir=False,
        use_IMU_flag=False,
        use_click_flag=False,
        use_LFP_flag=False,
        use_ripple_flag=False,
        only_flight_flag=True,
        no_3d_location=False,
        spike_count_flag=True,
        bin_width=0.10,
        split_flights_for_train='random',
        type_train_num=1,
        type_valid_num=1,
        special_id_count_thresh=3,
        only_use_special_id_4train_flag=False,
        shuffle_flag=True,
        time_shuffle_for_trajecotry_flag=False,
        simulate_place_cell_flag=False,
        split_each_trajectory_flag=True,
        window_size=None,
        stride_train=None,
        stride_test=None,
        batch_size_train=None,
        batch_size_test=None,
        session_names=list(session_names),
        parent_folder=parent_folder,
        max_acceleration=max_acceleration,
    )
