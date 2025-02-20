config = {
    'sr': 88200,
    'frame_rate': 60,
    'hidden_dim': 1024,
    'n_layers': 4, # if your own set to 8
    'num_heads': 4, # this should be 16 if its your own.
    'dropout': 0.0,
    'output_dim': 68, # if you trained your own, this should also be 61
    'input_dim': 256,
    'frame_size': 128, 
}
