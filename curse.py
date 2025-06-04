import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_blobs
import numpy as np

# Create figure and styling for plotting
fig, ax = plt.subplots(1, 1, figsize=(6, 3))
ax.set(xlabel='dimensions (m)', ylabel='log(dmax/dmin)', title='dmax/dmin vs. dimensionality')
line_styles = {0: 'ro-', 1: 'b^-', 2: 'gs-', 3: 'cv-'}

# Plot dmax/dmin ratio
# TODO: fill in valid test numbers
for idx, num_samples in enumerate([100, 1000, 10000]):
    # TODO: Fill in a valid feature range
    feature_range = range(1,100)
    ratios = []
    for num_features in feature_range:
        # TODO: Generate synthetic data using make_classification
        X,y = make_blobs(n_samples=num_samples, n_features=num_features)
        # TODO: Choose random query point from X
        query_point = np.random.choice(X[1])     
        # TODO: remove query pt from X so it isn't used in distance calculations

        # TODO: Calculate distances
        distances = []
        for x in X:
            if x[0]!=query_point:
                distances.append(np.linalg.norm(query_point-x[0]))
        ratio = np.max(distances) / np.min(distances)
        ratios.append(ratio)

    ax.plot(feature_range, np.log(ratios), line_styles[idx], label=f'N={num_samples:,}')

plt.legend()
plt.tight_layout()
plt.grid(True)
plt.savefig('dmaxdmin_graph.png')