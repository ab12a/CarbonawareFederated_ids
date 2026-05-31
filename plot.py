import matplotlib.pyplot as plt

accuracies = [0.7881, 0.7787, 0.7759, 0.7857]

rounds = range(1, len(accuracies)+1)

plt.plot(rounds, accuracies, marker='o')
plt.title("Federated Carbon-Aware Training Performance")
plt.xlabel("Training Round")
plt.ylabel("Detection Accuracy")
plt.grid(True)
plt.savefig("results/accuracy_plot.png")
plt.show()