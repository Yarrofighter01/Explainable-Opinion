import subprocess
import argparse
import sys
import os
import time

def run_shard(shard_idx, num_shards, base_model, output_dir, epochs=3):
    print(f"\n{'='*40}")
    print(f"Running Shard {shard_idx+1}/{num_shards}")
    print(f"{'='*40}\n")
    
    # For the first shard, use the base model.
    # For subsequent shards, use the output of the previous run.
    if shard_idx == 0:
        model_path = base_model
        print(f"Starting from base model: {model_path}")
    else:
        model_path = output_dir
        print(f"Continuing from previous checkpoint: {model_path}")
    
    cmd = [
        sys.executable, "src/train.py",
        "--dataset", "tweet_eval",
        "--model_path", model_path,
        "--output_dir", output_dir,
        "--epochs", str(epochs),
        "--num_shards", str(num_shards),
        "--shard_index", str(shard_idx)
    ]
    
    try:
        subprocess.check_call(cmd)
        print(f"Shard {shard_idx+1} completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error running shard {shard_idx}: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_shard", type=int, default=0, help="Shard index to start from")
    parser.add_argument("--num_shards", type=int, default=10, help="Total number of shards")
    parser.add_argument("--base_model", type=str, default="src/results", help="Initial base model")
    parser.add_argument("--output_dir", type=str, default="src/results_incremental", help="Output directory")
    
    args = parser.parse_args()
    
    for i in range(args.start_shard, args.num_shards):
        run_shard(i, args.num_shards, args.base_model, args.output_dir)
        # Optional: Cool down or gc
        time.sleep(2)

if __name__ == "__main__":
    main()
