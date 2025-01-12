import numpy as np
import random

def perceptron(features, labels, x0, gamma=0.1, eta=0.4, max_iterations=100, num_runs=3):
    # Implements the perceptron algorithm.

    all_results = []

    for run in range(num_runs):
        x = x0
        m = len(features)
        quality_history = []

        for k in range(max_iterations):
            #select a random index
            i = random.randint(0, m-1)
            a_i = features[i]
            y_i = labels[i]

            #update x based on the perceptron rule
            if y_i * np.dot(x, a_i) < 0:
                x = (1 - gamma) * x + eta * y_i * a_i

            #measure the quality of the prediction
            incorrect_count = 0
            for j in range(m):
                if labels[j] * np.dot(x, features[j]) < 0:
                    incorrect_count += 1

            quality_history.append(incorrect_count)

        all_results.append(quality_history)
    return all_results

# Example usage
features = np.array([
    [1.2, 2.4, 1.7],
    [-1.3, 1.0, 0.0],
    [0.0, 1.3, 0.4],
    [0.2, 0.5, 0.9],
    [-0.3, 1.1, -0.6],
    [-1.2, -0.7, -0.1]
])

labels = np.array([1, -1, 1, 1, -1, -1])
x0 = np.array([0.0, 0.0, 0.0])

results = perceptron(features, labels, x0)
for run, quality in enumerate(results):
    print(f"Run {run+1}: Incorrect classifications {quality}")