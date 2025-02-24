from pathlib import Path
import numpy as np

np.random.seed(1717)

output_dir = Path("output")
for subdir in output_dir.iterdir():
    if not subdir.is_dir():
        continue
    hidden_dim = None
    for npy_file in subdir.glob("*max_sim*pca_0.npy"):
        filename = npy_file.name
        config = np.load(npy_file, allow_pickle=True).item()
        first_direction = next(iter(config.values()))['first_direction']
        hidden_dim = first_direction.shape[0]
        dtype = first_direction.dtype
        
        random_direction_1 = np.random.randn(hidden_dim).astype(dtype)
        random_direction_1 /= np.linalg.norm(random_direction_1)
        random_direction_2 = np.random.randn(hidden_dim).astype(dtype)
        random_direction_2 /= np.linalg.norm(random_direction_2)
        
        # replace the second direction with a random direction
        for module_name in config.keys():
            config[module_name]['second_direction'] = random_direction_1
        
        # save the new config
        new_filename = filename.replace("pca_0", "dir_random")
        np.save(subdir / new_filename, config)
        
        # replace both directions with random directions
        for module_name in config.keys():
            config[module_name]['first_direction'] = random_direction_1
            config[module_name]['second_direction'] = random_direction_2
        
        # save the new config
        new_filename = "steering_config-xx-dir_random-dir_random.npy"
        np.save(subdir / new_filename, config)