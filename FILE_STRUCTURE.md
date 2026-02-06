# Project File Structure

All code files are organized in the `src/` directory.

> **Note:** The package was renamed from `code/` to `src/` because `code` is a
> built-in Python standard library module and shadows it, causing import errors.

## Created Files

1. **`src/envs.py`** - Contains both environment classes:
   - `SimpleGridEnv` - Simple navigation task
   - `KeyDoorBallEnv` - Multi-step key-door-ball task

2. **`src/preprocessing/`** - Preprocessing package:
   - **`preprocessing.py`** - Main preprocessing module:
     - `pre_process()` - Default preprocessing (RGB → grayscale 84×84)
     - `preprocess_grayscale()` - Convert RGB to grayscale
     - `preprocess_resize()` - Resize images
     - `preprocess_crop_and_resize()` - Crop borders and resize
     - `preprocess_normalize()` - Normalize pixel values
     - `get_preprocessed_observation_space()` - Get output shape
     - `visualize_preprocessing()` - Visualize original vs preprocessed images
     - `test_preprocessing_with_visualization()` - Run tests and generate graphs
   - **`__init__.py`** - Package initialization

3. **`src/networks/`** - Neural network package:
   - **`networks.py`** - Network architectures:
     - `obs_to_tensor()` - Observation → tensor conversion helper
     - `CNNFeatureExtractor` - CNN backbone (DQN Nature architecture)
     - `PolicyNetwork` - Policy network for REINFORCE
   - **`test_networks.py`** - Unit tests for networks
   - **`__init__.py`** - Package initialization

4. **`src/reinforce/`** - REINFORCE algorithm package:
   - **`reinforce.py`** - Implementation:
     - `compute_returns()` - Discounted returns
     - `train_reinforce()` - Training loop with batched updates
     - `TrainedAgent` - Wrapper for evaluation
   - **`__init__.py`** - Package initialization

5. **`src/utils.py`** - Shared utilities:
   - `plot_training_history()` - Plot reward, steps, entropy curves
   - `evaluate_agent()` - Evaluate agent over multiple episodes
   - `print_evaluation_results()` - Format and print evaluation results
   - `record_video()` - Record agent gameplay as GIF

6. **`src/experiments/`** - Experiment scripts:
   - **`run_simplegrid.py`** - REINFORCE on SimpleGridEnv
   - **`__init__.py`** - Package initialization

7. **`src/__init__.py`** - Main package initialization

## Still Planned

- **`src/dqn/`** - Deep Q-Network implementation
- **`src/experiments/run_keydoorball.py`** - REINFORCE/DQN on KeyDoorBallEnv

## Notes
- All files should be modular and importable
- Use PyTorch for neural networks (no external RL libraries allowed)
- Follow the structure from `solution.ipynb` but organize into separate files
- At the end, merge all code back into the notebook for submission
