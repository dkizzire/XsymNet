import json
import matplotlib.pyplot as plt
import re
import os

# === 1. Load JSON data ===
with open("logit_results_P1_fe2o3.json", "r") as f:
    data = json.load(f)

# === 2. Extract numeric part (e.g. 100 from nom_208518_100-4) ===
def extract_numeric_id(key):
    match = re.search(r'_(\d+)-', key)
    return int(match.group(1)) if match else None

# === 3. Sort keys by extracted numeric ID ===
sorted_items = sorted(data.items(), key=lambda item: extract_numeric_id(item[0]))

# === 4. Extract all logit and subgroup info per index
num_logits = len(next(iter(data.values()))["logits"])  # Assume all entries have same length

# Create output folder
output_folder = "logit_plots"
os.makedirs(output_folder, exist_ok=True)

for i in range(num_logits):
    x_vals = []
    logits = []
    subgroups = []

    for key, entry in sorted_items:
        x_val = extract_numeric_id(key)
        logit = entry["logits"][i]
        subgroup = int(entry["subgroups"][i])  # Convert "0005" -> 5

        x_vals.append(x_val)
        logits.append(logit)
        subgroups.append(subgroup)

    # === 5. Plotting ===
    plt.figure(figsize=(24, 6))
    plt.plot(x_vals, logits, marker='o', label=f'Logit {i}')

    for x, y, sg in zip(x_vals, logits, subgroups):
        plt.text(x, y, str(sg), fontsize=9, ha='center', va='bottom')

    plt.xlabel("Temperature (K)")
    plt.ylabel(f"Logit {i} Value")
    plt.title(f"Evolution of Logit {i} Over Temperature ID")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()

    # === 6. Save each plot ===
    plt.savefig(f"{output_folder}/logit_{i}_plot.png", dpi=300)
    plt.close()
