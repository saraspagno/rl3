"""
Preprocessing utilities for MiniGrid observations.

This module provides preprocessing functions to convert raw RGB observations
from MiniGrid environments into processed observations suitable for neural networks.
"""

import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Add parent directory to path to import environments
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs import SimpleGridEnv, KeyDoorBallEnv


def preprocess_grayscale(img):
    """
    Preprocess raw RGB observation from the environment by converting to grayscale.

    Args:
        img: RGB image array of shape (320, 320, 3) with dtype uint8

    Returns:
        Grayscale image array of shape (320, 320, 1) with dtype uint8
    """
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return np.expand_dims(gray, axis=-1)
    elif len(img.shape) == 2:
        # Already grayscale, just add channel dimension
        return np.expand_dims(img, axis=-1)
    else:
        # Already has single channel
        return img


def preprocess_resize(img, target_size=(64, 64)):
    """
    Preprocess observation by resizing to a smaller size.

    Args:
        img: RGB image array of shape (320, 320, 3) or grayscale (320, 320, 1)
        target_size: Target size tuple (height, width)

    Returns:
        Resized image array with shape (target_size[0], target_size[1], channels)
    """
    if len(img.shape) == 3:
        if img.shape[2] == 1:
            # Grayscale with channel dimension
            resized = cv2.resize(img[:, :, 0], target_size, interpolation=cv2.INTER_AREA)
            return np.expand_dims(resized, axis=-1)
        else:
            # RGB
            resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
            return resized
    else:
        # 2D grayscale
        resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        return np.expand_dims(resized, axis=-1)


def preprocess_crop_and_resize(img, crop_size=(280, 280), target_size=(64, 64)):
    """
    Preprocess observation by cropping borders and resizing.

    Args:
        img: RGB image array of shape (320, 320, 3)
        crop_size: Size to crop to (height, width) - removes borders
        target_size: Target size after resize (height, width)

    Returns:
        Processed image array
    """
    h, w = img.shape[:2]
    crop_h, crop_w = crop_size
    
    # Calculate crop offsets (center crop)
    start_h = (h - crop_h) // 2
    start_w = (w - crop_w) // 2
    
    # Crop
    cropped = img[start_h:start_h+crop_h, start_w:start_w+crop_w]
    
    # Resize
    return preprocess_resize(cropped, target_size)


def preprocess_normalize(img, mean=127.5, std=127.5):
    """
    Normalize image to [-1, 1] or [0, 1] range.

    Args:
        img: Image array (any shape)
        mean: Mean value for normalization
        std: Standard deviation for normalization

    Returns:
        Normalized image array (float32)
    """
    img_float = img.astype(np.float32)
    normalized = (img_float - mean) / std
    return normalized


def get_preprocessed_observation_space(preprocess_func, original_shape=(320, 320, 3)):
    """
    Get the observation space shape after preprocessing.

    Args:
        preprocess_func: Preprocessing function
        original_shape: Original observation shape

    Returns:
        Observation space shape tuple
    """
    # Create dummy input
    dummy_input = np.zeros(original_shape, dtype=np.uint8)
    processed = preprocess_func(dummy_input)
    return processed.shape


# Default preprocessing function (can be imported and used directly)
def pre_process(img):
    """
    Default preprocessing function:
        1. Crop the outer wall frame (1 cell from each edge of the 10×10 grid)
        2. Resize to 84×84
        3. Keep RGB (3 channels) — colour is critical for distinguishing
           the red agent from the green goal.

    Args:
        img: RGB image from MiniGrid renderer, shape (H, W, 3).
             H = W = grid_size * tile_size  (e.g. 100 for tile_size=10).

    Returns:
        RGB image (84, 84, 3) with values 0–255 (uint8).
    """
    h, w = img.shape[:2]

    # 1. Crop outer wall frame: 1/10 margin on each side (for 10×10 grid)
    margin = h // 10
    cropped = img[margin : h - margin, margin : w - margin]

    # 2. Resize to 84×84 (keep RGB)
    #    INTER_NEAREST is fastest; fine for a tiny 80→84 upscale on pixel-art
    resized = cv2.resize(cropped, (84, 84), interpolation=cv2.INTER_NEAREST)

    return resized  # (84, 84, 3)


# =============================================================================
# Visualization Functions
# =============================================================================

def visualize_preprocessing(img, preprocess_func, title="Preprocessed", save_path=None):
    """
    Visualize original and preprocessed images side by side.

    Args:
        img: Original RGB image
        preprocess_func: Preprocessing function to apply
        title: Title for the preprocessed image
        save_path: Path to save the visualization (optional)

    Returns:
        Processed image
    """
    processed = preprocess_func(img)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original image
    axes[0].imshow(img)
    axes[0].set_title("Original RGB Image")
    axes[0].axis("off")
    
    # Processed image
    if len(processed.shape) == 3 and processed.shape[2] == 1:
        # Grayscale with channel dimension
        axes[1].imshow(processed.squeeze(), cmap="gray")
    elif len(processed.shape) == 2:
        # 2D grayscale
        axes[1].imshow(processed, cmap="gray")
    else:
        # RGB or other
        axes[1].imshow(processed)
    
    axes[1].set_title(f"{title}\nShape: {processed.shape}, dtype: {processed.dtype}")
    axes[1].axis("off")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    
    plt.close()
    
    return processed


def visualize_all_preprocessing_methods(img, output_dir=None, prefix=""):
    """
    Visualize all preprocessing methods and save comparison plots.

    Args:
        img: Original RGB image
        output_dir: Directory to save visualizations (defaults to preprocessing folder)
        prefix: Prefix for output filenames (e.g., "simplegrid" or "keydoorball")
    """
    if output_dir is None:
        # Get the directory where this file is located
        output_dir = os.path.dirname(os.path.abspath(__file__))
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Define preprocessing methods to test
    methods = [
        ("Grayscale", preprocess_grayscale),
        ("Resize (64x64)", lambda x: preprocess_resize(x, target_size=(64, 64))),
        ("Crop & Resize", lambda x: preprocess_crop_and_resize(x, crop_size=(280, 280), target_size=(64, 64))),
    ]
    
    # Create a large comparison figure
    n_methods = len(methods)
    fig, axes = plt.subplots(2, n_methods + 1, figsize=(4 * (n_methods + 1), 8))
    
    # Original image (spans both rows)
    axes[0, 0].imshow(img)
    axes[0, 0].set_title("Original RGB\n(320, 320, 3)", fontsize=10)
    axes[0, 0].axis("off")
    axes[1, 0].axis("off")  # Empty cell below
    
    # Process each method
    for idx, (name, func) in enumerate(methods):
        col = idx + 1
        processed = func(img)
        
        # Top row: processed image
        if len(processed.shape) == 3 and processed.shape[2] == 1:
            axes[0, col].imshow(processed.squeeze(), cmap="gray")
        elif len(processed.shape) == 2:
            axes[0, col].imshow(processed, cmap="gray")
        else:
            axes[0, col].imshow(processed)
        
        axes[0, col].set_title(f"{name}\n{processed.shape}", fontsize=10)
        axes[0, col].axis("off")
        
        # Bottom row: histogram
        if len(processed.shape) == 3:
            data = processed.flatten()
        else:
            data = processed.flatten()
        
        axes[1, col].hist(data, bins=50, alpha=0.7, edgecolor='black')
        axes[1, col].set_title(f"Pixel Distribution", fontsize=9)
        axes[1, col].set_xlabel("Pixel Value")
        axes[1, col].set_ylabel("Frequency")
    
    title = "Preprocessing Methods Comparison"
    if prefix:
        title = f"{prefix.title()} - {title}"
    plt.suptitle(title, fontsize=14, y=0.995)
    plt.tight_layout()
    
    comparison_filename = "preprocessing_comparison.png"
    if prefix:
        comparison_filename = f"{prefix}_preprocessing_comparison.png"
    comparison_path = os.path.join(output_dir, comparison_filename)
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison visualization to: {comparison_path}")
    plt.close()
    
    # Save individual visualizations
    for name, func in methods:
        safe_name = name.lower().replace(" ", "_").replace("&", "and")
        filename = f"preprocessing_{safe_name}.png"
        if prefix:
            filename = f"{prefix}_{filename}"
        save_path = os.path.join(output_dir, filename)
        visualize_preprocessing(img, func, title=name, save_path=save_path)


# =============================================================================
# Tests
# =============================================================================

def test_preprocessing_with_visualization():
    """Test preprocessing functions using actual environment images and generate visualization graphs."""
    print("Testing preprocessing functions with actual environment images...")
    
    # Get output directory (same folder as this file)
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Initialize environments to get real observations
    print("\n1. Initializing environments...")
    simple_env = SimpleGridEnv(max_steps=100, preprocess=lambda x: x)  # No preprocessing to get raw RGB
    keydoor_env = KeyDoorBallEnv(max_steps=100, preprocess=lambda x: x)  # No preprocessing to get raw RGB
    
    # Get observations from environments
    print("2. Collecting observations from environments...")
    simple_obs, _ = simple_env.reset(seed=42)
    simple_rgb = simple_env.render()  # Get raw RGB frame
    
    keydoor_obs, _ = keydoor_env.reset(seed=42)
    keydoor_rgb = keydoor_env.render()  # Get raw RGB frame
    
    print(f"   SimpleGridEnv RGB shape: {simple_rgb.shape}")
    print(f"   KeyDoorBallEnv RGB shape: {keydoor_rgb.shape}")
    
    # Test preprocessing functions
    print("\n3. Testing preprocessing functions...")
    
    # Test grayscale preprocessing
    print("   Testing preprocess_grayscale...")
    gray = preprocess_grayscale(simple_rgb)
    assert gray.shape == (320, 320, 1), f"Expected shape (320, 320, 1), got {gray.shape}"
    assert gray.dtype == np.uint8, f"Expected dtype uint8, got {gray.dtype}"
    assert gray.min() >= 0 and gray.max() <= 255, "Values should be in [0, 255]"
    print("   ✓ Grayscale preprocessing works correctly")
    
    # Test default pre_process function
    print("   Testing pre_process (default function)...")
    processed = pre_process(simple_rgb)
    assert processed.shape == (84, 84, 3), f"Expected shape (84, 84, 3), got {processed.shape}"
    assert processed.dtype == np.uint8, f"Expected dtype uint8, got {processed.dtype}"
    print("   ✓ Default pre_process function works correctly")
    
    # Test resize preprocessing
    print("   Testing preprocess_resize...")
    resized = preprocess_resize(simple_rgb, target_size=(64, 64))
    assert resized.shape == (64, 64, 3), f"Expected shape (64, 64, 3), got {resized.shape}"
    print("   ✓ Resize preprocessing works correctly")
    
    # Test resize on grayscale
    gray_resized = preprocess_resize(gray, target_size=(64, 64))
    assert gray_resized.shape == (64, 64, 1), f"Expected shape (64, 64, 1), got {gray_resized.shape}"
    print("   ✓ Grayscale resize preprocessing works correctly")
    
    # Test crop and resize
    print("   Testing preprocess_crop_and_resize...")
    cropped_resized = preprocess_crop_and_resize(simple_rgb, crop_size=(280, 280), target_size=(64, 64))
    assert cropped_resized.shape == (64, 64, 3), f"Expected shape (64, 64, 3), got {cropped_resized.shape}"
    print("   ✓ Crop and resize preprocessing works correctly")
    
    # Test normalize
    print("   Testing preprocess_normalize...")
    normalized = preprocess_normalize(gray)
    assert normalized.dtype == np.float32, f"Expected dtype float32, got {normalized.dtype}"
    assert normalized.min() >= -1.0 and normalized.max() <= 1.0, "Values should be in [-1, 1]"
    print("   ✓ Normalization works correctly")
    
    # Test observation space helper
    print("   Testing get_preprocessed_observation_space...")
    obs_shape = get_preprocessed_observation_space(pre_process)
    assert obs_shape == (84, 84, 3), f"Expected shape (84, 84, 3), got {obs_shape}"
    print("   ✓ Observation space helper works correctly")
    
    print("\n" + "="*60)
    print("All preprocessing tests passed! ✓")
    print("="*60)
    
    # Generate visualizations with actual environment images
    print("\n4. Generating preprocessing visualizations...")
    print("="*60)
    
    # Create a single visualization with both environments
    print("\n   Generating preprocessing visualization (both environments)...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # SimpleGridEnv - Original
    axes[0, 0].imshow(simple_rgb)
    axes[0, 0].set_title("SimpleGridEnv - Original RGB", fontsize=12, fontweight='bold')
    axes[0, 0].axis("off")
    
    # SimpleGridEnv - Preprocessed
    simple_processed = pre_process(simple_rgb)
    axes[0, 1].imshow(simple_processed)
    axes[0, 1].set_title(f"SimpleGridEnv - Preprocessed (RGB 84x84)\nShape: {simple_processed.shape}", fontsize=12, fontweight='bold')
    axes[0, 1].axis("off")
    
    # KeyDoorBallEnv - Original
    axes[1, 0].imshow(keydoor_rgb)
    axes[1, 0].set_title("KeyDoorBallEnv - Original RGB", fontsize=12, fontweight='bold')
    axes[1, 0].axis("off")
    
    # KeyDoorBallEnv - Preprocessed
    keydoor_processed = pre_process(keydoor_rgb)
    axes[1, 1].imshow(keydoor_processed)
    axes[1, 1].set_title(f"KeyDoorBallEnv - Preprocessed (RGB 84x84)\nShape: {keydoor_processed.shape}", fontsize=12, fontweight='bold')
    axes[1, 1].axis("off")
    
    plt.suptitle("Preprocessing: Before and After", fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "preprocessing_before_after.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   Saved visualization to: {output_path}")
    plt.close()
    
    print("\n" + "="*60)
    print("Visualization tests completed! ✓")
    print(f"Check the preprocessing folder for saved image: {output_path}")
    print("="*60)


if __name__ == "__main__":
    # Run tests with visualization when script is executed directly
    test_preprocessing_with_visualization()
