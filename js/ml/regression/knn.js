/**
 * K-Nearest Neighbours Regressor.
 * Predicts the (weighted) average of the K closest training samples
 * using Euclidean distance.
 * @module regression/knn
 */

/**
 * @class KNNRegressor
 * @description Instance-based regression using K nearest neighbours.
 */
export class KNNRegressor {
    /**
     * @param {Object}  params              - Configuration.
     * @param {number}  params.nNeighbors    - Number of neighbours (default 5).
     * @param {string}  params.weights       - 'uniform' or 'distance' (default 'uniform').
     */
    constructor(params = {}) {
        /** @type {number} */
        this.nNeighbors = params.nNeighbors ?? 5;
        /** @type {string} */
        this.weights = params.weights ?? 'uniform';
        /** @type {number[][]|null} Stored training features */
        this._X = null;
        /** @type {number[]|null} Stored training targets */
        this._y = null;
        /** @type {number} */
        this.nFeatures = 0;
    }

    /**
     * Compute Euclidean distance between two vectors.
     * @param {number[]} a
     * @param {number[]} b
     * @returns {number}
     */
    static _euclidean(a, b) {
        let sum = 0;
        for (let i = 0; i < a.length; i++) {
            const d = a[i] - b[i];
            sum += d * d;
        }
        return Math.sqrt(sum);
    }

    /**
     * Store training data (lazy learner).
     * @param {number[][]} X - 2D array [nSamples, nFeatures].
     * @param {number[]}   y - 1D array [nSamples].
     * @returns {KNNRegressor}
     */
    fit(X, y) {
        if (!X || !y || X.length === 0 || y.length === 0) {
            throw new Error('Training data must not be empty.');
        }
        if (X.length !== y.length) {
            throw new Error('X and y must have the same number of samples.');
        }

        this.nFeatures = X[0].length;
        // Store immutable copies
        this._X = X.map(row => [...row]);
        this._y = [...y];

        return this;
    }

    /**
     * Find K nearest neighbours and predict.
     * @param {number[][]} X - 2D array [nSamples, nFeatures].
     * @returns {number[]}
     */
    predict(X) {
        if (!this._X || !this._y) {
            throw new Error('Model has not been fitted yet. Call fit() first.');
        }
        if (!X || X.length === 0) {
            return [];
        }

        const k = Math.min(this.nNeighbors, this._X.length);

        return X.map(sample => {
            // Compute distances to all training points
            const distances = this._X.map((trainRow, idx) => ({
                index: idx,
                dist: KNNRegressor._euclidean(sample, trainRow),
            }));

            // Sort ascending by distance and take top K
            distances.sort((a, b) => a.dist - b.dist);
            const neighbours = distances.slice(0, k);

            if (this.weights === 'distance') {
                return this._distanceWeightedAverage(neighbours);
            }
            // Uniform: simple average
            const sum = neighbours.reduce((acc, nb) => acc + this._y[nb.index], 0);
            return sum / k;
        });
    }

    /**
     * Compute distance-weighted average of neighbour targets.
     * Falls back to uniform if all distances are zero.
     * @param {{ index: number, dist: number }[]} neighbours
     * @returns {number}
     */
    _distanceWeightedAverage(neighbours) {
        // Inverse-distance weighting: w_i = 1 / d_i
        const allZero = neighbours.every(nb => nb.dist === 0);
        if (allZero) {
            const sum = neighbours.reduce((acc, nb) => acc + this._y[nb.index], 0);
            return sum / neighbours.length;
        }

        let weightSum = 0;
        let valueSum = 0;
        for (const nb of neighbours) {
            const w = nb.dist === 0 ? 1e12 : 1 / nb.dist;
            weightSum += w;
            valueSum += w * this._y[nb.index];
        }
        return valueSum / weightSum;
    }

    /**
     * Return model parameters.
     * @returns {{ nNeighbors: number, weights: string }}
     */
    getParams() {
        return {
            nNeighbors: this.nNeighbors,
            weights: this.weights,
        };
    }

    /**
     * KNN does not inherently provide feature importances.
     * @returns {null}
     */
    getFeatureImportance() {
        return null;
    }
}
