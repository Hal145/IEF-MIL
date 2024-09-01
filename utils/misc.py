import os
import re
import shutil

def copy_images_with_pattern(source_dir, dest_dir, pattern):
  """
  Copies images from source_dir to subdirectories in dest_dir named after the matched pattern in the image name.
  Creates subdirectories in dest_dir if they don't exist.

  Args:
      source_dir (str): Path to the source directory containing images.
      dest_dir (str): Path to the destination directory (parent directory).
      pattern (str): Regular expression pattern to match in image names.
  """

  # Compile the pattern for efficient matching
  pattern_regex = re.compile(pattern)

  # Get all files in the source directory
  files = os.listdir(source_dir)

  # Loop through files and copy images with matching pattern
  for filename in files:
    match = pattern_regex.search(filename)
    if match:
      # Extract the matched pattern as the subdirectory name
      subdirectory_name = match.group()

      # Construct the subdirectory path within dest_dir
      subdirectory_path = os.path.join(dest_dir, subdirectory_name)

      # Create the subdirectory if it doesn't exist (handling potential errors)
      try:
        os.makedirs(subdirectory_path, exist_ok=True)
      except OSError as e:
        print(f"Error creating directory '{subdirectory_path}': {e}")
        continue  # Skip to next file if directory creation fails

      # Construct full paths for source and destination files
      source_path = os.path.join(source_dir, filename)
      dest_path = os.path.join(subdirectory_path, filename)

      # Check if the file is an image (optional, based on your needs)
      if os.path.isfile(source_path) and filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
        try:
          # Copy the image file
          shutil.copy2(source_path, dest_path)
          print(f"Copied '{filename}' to '{subdirectory_path}'")
        except OSError as e:
          print(f"Error copying '{filename}': {e}")

if __name__ == "__main__":
  source_dir = "/media/nfs/HL_dataset/HEATMAPS/heatmap_production_results/HEATMAP_OUTPUT_DSMIL_mean_5x+10x+20x_EGE_hipkon/sampled_patches/label_subtype_4_pred_3/topk_high_attention"  # Replace with your source directory path
  dest_dir = "/media/nfs/send/LD"  # Replace with your desired destination path
  pattern = "capa-ld-12"  # Regular expression pattern (replace with your desired pattern)

  copy_images_with_pattern(source_dir, dest_dir, pattern)
