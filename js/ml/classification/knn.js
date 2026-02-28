/**
 * K-Nearest Neighbors classifier.
 * Supports uniform and distance-based weighting.
 * @module classification/knn
 */

/**
 * @class KNNClassifier
 */
export class KNNClassifier {
    /**
     * @param {Object} params
     * @param {number} [params.nNeighbors=5] - Number of neighbors
     * @param {string} [params.weights='uniform'] - Weighting scheme: 'uniform' or 'distance'
     */
    constructor({ nNeighbors = 5, weights = 'uniform' } = {}) {
        if (!['uniform', 'distance'].includes(weights)) {
            throw new Error("weights must be 'uniform' or 'distance'");
        }
        this.nNeighbors = nNeighbors;
        this.weights = weights;
        this.XTrain = null;
        this.yTrain = null;
        this.classes = null;
        this.nFeatures = null;
    }

    /**
     * Compute Euclidean distance between two vectors.
     * @private
     * @param {number[]} a
     * @param {number[]} b
     * @returns {number}
     */
    _euclideanDistance(a, b) {
        let sum = 0;
        for (let i = 0; i < a.length; i++) {
            const diff = a[i] - b[i];
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }

    /**
     * Find the K nearest neighbors for a single sample.
     * @private
     * @param {number[]} row
     * @returns {{ index: number, distance: number }[]}
     */
    _findNeighbors(row) {
        const distances = this.XTrain.map((trainRow, idx) => ({
            index: idx,
            distance: this._euclideanDistance(row, trainRow)
        }));

        distances.sort((a, b) => a.distance - b.distance);
        return distances.slice(0, this.nNeighbors);
    }

    /**
     * Fit the model (stores training data).
     * @param {number[][]} X - Feature matrix
     * @param {number[]} y - Class labels
     * @returns {KNNClassifier} this
     */
    fit(X, y) {
        if (!X || !X.length || !y || !y.length) {
            throw new Error('X and y must be non-empty arrays');
        }
        if (X.length !== y.length) {
            throw new Error('X and y must have the same number of samples');
        }

        this.XTrain = X.map(row => [...row]);
        this.yTrain = [...y];
        this.classes = [...new Set(y)].sort((a, b) => a - b);
        this.nFeatures = X[0].length;

        if (this.nNeighbors > X.length) {
            this.nNeighbors = X.length;
        }

        return this;
    }

    /**
     * Predict class probabilities.
     * @param {number[][]} X - Feature matrix
     * @returns {number[][]} Probabilities for each class per sample
     */
    predictProba(X) {
        if (!this.XTrain) {
            throw new Error('Model has not been fitted yet');
        }

        return X.map(row => {
            const neighbors = this._findNeighbors(row);
            const classWeights = {};
            for (const cls of this.classes) {
                classWeights[cls] = 0;
            }

            if (this.weights === 'uniform') {
                for (const neighbor of neighbors) {
                    const label = this.yTrain[neighbor.index];
                    classWeights[label] += 1;
                }
            } else {
                for (const neighbor of neighbors) {
                    const label = this.yTrain[neighbor.index];
                    const weight = neighbor.distance === 0 ? 1e10 : 1 / neighbor.distance;
                    classWeights[label] += weight;
                }
            }

            const totalWeight = Object.values(classWeights).reduce((a, b) => a + b, 0);
            return this.classes.map(cls =>
                totalWeight > 0 ? classWeights[cls] / totalWeight : 1 / this.classes.length
            );
        });
    }

    /**
     * Predict class labels.
     * @param {number[][]} X - Feature matrix
     * @returns {number[]} Predicted labels
     */
    predict(X) {
        const proba = this.predictProba(X);
        return proba.map(probs => {
            const maxIdx = probs.indexOf(Math.max(...probs));
            return this.classes[maxIdx];
        });
    }

    /** @returns {Object} */
    getParams() {
        return {
            nNeighbors: this.nNeighbors,
            weights: this.weights
        };
    }

    /**
     * KNN does not compute feature importances.
     * @returns {null}
     */
    getFeatureImportance() {
        return null;
    }
}
