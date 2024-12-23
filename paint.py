import os

import matplotlib.pyplot as plt
import numpy as np
import re

def read_log_file(file_path):
    episodes = []
    losses = []
    avg_scores = []
    avg_len_snake = []
    max_len_snake = []

    print("Reading file:", file_path)  # Debug print

    with open(file_path, 'r') as f:
        # Print first few lines of the file to see actual format
        print("First few lines of the file:")
        for i, line in enumerate(f):
            if i < 3:  # Print first 3 lines
                print(f"Line {i+1}: {line.strip()}")

    # Now read the file for real
    with open(file_path, 'r') as f:
        for line in f:
            try:
                # Split by comma and clean up whitespace
                parts = [part.strip() for part in line.split(',')]

                # Extract values using more lenient parsing
                ep = int(parts[0].split(':')[1])
                loss = float(parts[1].split(':')[1])
                avg_score = float(parts[2].split(':')[1])
                avg_len = float(parts[3].split(':')[1])
                max_len = float(parts[4].split(':')[1].split('Time')[0])

                episodes.append(ep)
                losses.append(loss)
                avg_scores.append(avg_score)
                avg_len_snake.append(avg_len)
                max_len_snake.append(max_len)

            except Exception as e:
                print(f"Error parsing line: {line.strip()}")
                print(f"Error details: {str(e)}")
                continue

    if not episodes:
        raise ValueError("No valid data found in log file. Please check the file format.")

    print(f"Successfully parsed {len(episodes)} episodes")  # Debug print
    return episodes, losses, avg_scores, avg_len_snake, max_len_snake
def plot_metrics(input_dir, max_episodes=5000):
    """
    Plot metrics with optional episode limit
    Args:
        input_dir: Directory containing log.txt
        max_episodes: Maximum number of episodes to plot (None for all episodes)
    """
    # Read data from log file
    log_path = os.path.join(input_dir, "log.txt")
    episodes, losses, avg_scores, avg_len_snake, max_len_snake = read_log_file(log_path)

    # If max_episodes is specified, limit the data
    if max_episodes is not None:
        # Find the index where episodes exceed max_episodes
        end_idx = next((i for i, ep in enumerate(episodes) if ep > max_episodes), len(episodes))
        episodes = episodes[:end_idx]
        avg_scores = avg_scores[:end_idx]
        avg_len_snake = avg_len_snake[:end_idx]
        max_len_snake = max_len_snake[:end_idx]

    # Figure 1: Average Scores
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, avg_scores, label="Avg score")
    plt.legend()
    plt.ylabel('Score')
    plt.xlabel('Episodes #')
    plt.title(f'Average Scores over Episodes (First {max_episodes if max_episodes else "all"} episodes)')
    plt.grid(True)
    plt.savefig(os.path.join(input_dir, "scores.png"), bbox_inches='tight')
    plt.close()

    # Figure 2: Snake Lengths
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, avg_len_snake, label="Avg Len of Snake")
    plt.plot(episodes, max_len_snake, label="Max Len of Snake")
    plt.legend()
    plt.ylabel('Length of Snake')
    plt.xlabel('Episodes #')
    plt.title(f'Snake Lengths over Episodes (First {max_episodes if max_episodes else "all"} episodes)')
    plt.grid(True)
    plt.savefig(os.path.join(input_dir, "lengths.png"), bbox_inches='tight')
    plt.close()

    # Figure 3: Histogram of max lengths
    plt.figure(figsize=(10, 6))
    n, bins, patches = plt.hist(max_len_snake, 45, density=1, facecolor='green', alpha=0.75)
    mu = round(np.mean(max_len_snake), 2)
    sigma = round(np.std(max_len_snake), 2)
    median = round(np.median(max_len_snake), 2)
    print('mu: ', mu, ', sigma: ', sigma, ', median: ', median)
    plt.xlabel(f'Max.Lengths, mu = {mu:.2f}, sigma={sigma:.2f}, median: {median:.2f}')
    plt.ylabel('Probability')
    plt.title(f'Histogram of Max.Lengths (First {max_episodes if max_episodes else "all"} episodes)')
    plt.axis([4, 44, 0, 0.15])
    plt.grid(True)
    plt.savefig(os.path.join(input_dir, "max_length_hist.png"), bbox_inches='tight')
    plt.close()

# Usage
input_directory = "/Users/zhanghanxing/Desktop/work/abv/Snakeee/dir_chk_5"
plot_metrics(input_directory)