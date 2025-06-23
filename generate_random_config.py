from pathlib import Path
import numpy as np

from configs import MAX_SIM_DIR_ID

np.random.seed(1717)

for model_dir, direction_id in MAX_SIM_DIR_ID.items():
    print(f"Processing model: {model_dir}, direction_id: {direction_id}")
    output_dir = Path("output") / model_dir.split("/")[1]
    hidden_dim = None
    for steering_config_file in output_dir.glob("steering_config-*.npy"):
        if (
            direction_id not in steering_config_file.stem
            or "pca_0" not in steering_config_file.stem
        ):
            continue

        filename = steering_config_file.name
        config = np.load(steering_config_file, allow_pickle=True).item()
        first_direction = next(iter(config.values()))["first_direction"]
        hidden_dim = first_direction.shape[0]
        dtype = first_direction.dtype

        random_direction_1 = np.random.randn(hidden_dim).astype(dtype)
        random_direction_1 /= np.linalg.norm(random_direction_1)
        random_direction_2 = np.random.randn(hidden_dim).astype(dtype)
        random_direction_2 /= np.linalg.norm(random_direction_2)

        # replace the second direction with a random direction
        for module_name in config.keys():
            config[module_name]["second_direction"] = random_direction_1

        # save the new config
        new_filename = filename.replace("pca_0", "dir_random")
        print(f"\tSaving new steering config: {new_filename}")
        np.save(output_dir / new_filename, config)

        # replace both directions with random directions
        for module_name in config.keys():
            config[module_name]["first_direction"] = random_direction_1
            config[module_name]["second_direction"] = random_direction_2

        # save the new config
        new_filename = "steering_config-xx-dir_random-dir_random.npy"
        print(f"\tSaving new steering config: {new_filename}")
        np.save(output_dir / new_filename, config)
