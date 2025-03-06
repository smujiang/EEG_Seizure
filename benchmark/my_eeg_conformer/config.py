
# Default configuration for the my_eeg_gpt benchmark
default_config = {
    # model parameters
    "batch_size": 72,
    "n_epochs": 2000,
    "c_dim": 2,
    "lr": 0.0002,
    "b1": 0.5,
    "b2": 0.999,
    "dimension": (190, 50),
    "start_epoch": 0,
    "data_root": "/Data/strict_TE/",
    "log_write_path": "./results/log_subject.txt",
    "window_size_sec" : 25,
    "fs" : 256,
    "montage" : "UNIPOLAR"
}


class Config():
    def __init__(self, default_config, **kwargs):
        for key, value in default_config.items():
            setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)