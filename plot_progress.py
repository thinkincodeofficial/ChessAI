"""Turn training_log.csv into a chart for the video:  python plot_progress.py"""
import csv
import matplotlib.pyplot as plt

rows = list(csv.DictReader(open("training_log.csv")))
gen = [int(r["generation"]) for r in rows]
score = [(int(r["wins"]) + 0.5 * int(r["draws"])) / (int(r["wins"]) + int(r["draws"]) + int(r["losses"])) * 100
         for r in rows]
loss = [float(r["loss"]) for r in rows]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
ax1.plot(gen, score, marker="o", color="#2a9d8f", linewidth=2)
ax1.axhline(50, color="gray", linestyle="--", linewidth=1)
ax1.set(title="Score vs MaterialBot (win = 1, draw = ½)", xlabel="Generation", ylabel="Score %", ylim=(0, 100))
ax2.plot(gen, loss, color="#e76f51", linewidth=2)
ax2.set(title="Training loss", xlabel="Generation", ylabel="MSE")
for ax in (ax1, ax2):
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("training_progress.png", dpi=150)
print("Saved training_progress.png")
