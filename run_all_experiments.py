import subprocess
import sys
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Run all experiments')
    parser.add_argument('--project', type=str, default='prpd', help='WandB project name')
    return parser.parse_args()

# Configuration
models = ['covamnet', 'protonet', 'cosine', 'matchingnet', 'relationnet']
shots = [1, 5]
samples_list = [12, 60, None]  # None means all samples
lambda_center = 0

total_experiments = len(models) * len(shots) * len(samples_list)
current_experiment = 0

args = get_args()
print(f"=" * 80)
print(f"Configuration:")
print(f"  Models: {models} ({len(models)})")
print(f"  Shots: {shots} ({len(shots)})")
print(f"  Samples: {samples_list} ({len(samples_list)})")
print(f"  Loss: Paper-standard (auto-selected per model)")
print(f"  Lambda Center: {lambda_center} (fixed)")
print(f"  Total: {len(models)} × {len(shots)} × {len(samples_list)} = {total_experiments}")
print(f"=" * 80)
print(f"Starting {total_experiments} experiments on project '{args.project}'...")

for model in models:
    for shot in shots:
        for samples in samples_list:
            current_experiment += 1
            print(f"\n[{current_experiment}/{total_experiments}] Running: Model={model}, Shot={shot}, Samples={samples if samples else 'All'}")
            
            cmd = [sys.executable, 'main.py', 
                   '--model', model, 
                   '--shot_num', str(shot), 
                   '--loss', 'contrastive',  # Default, auto-overridden for relationnet
                   '--lambda_center', str(lambda_center),
                   '--mode', 'train',
                   '--project', args.project]
            
            if samples is not None:
                cmd.extend(['--training_samples', str(samples)])
            
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error running experiment: {e}")
                # Decide whether to continue or stop. Continuing is usually safer for long runs.
                continue

print("\nAll experiments completed.")
