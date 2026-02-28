/**
 * Logistic Regression classifier.
 * Binary classification via sigmoid + gradient descent.
 * Multi-class via one-vs-rest strategy.
 * @module classification/logistic
 */

/**
 * @class LogisticRegression
 */
export class LogisticRegression {
    /**
     * @param {Object} params
     * @param {number} [params.learningRate=0.01] - Step size for gradient descent
     * @param {number} [params.maxIter=1000] - Maximum iterations
     * @param {number} [params.tol=1e-4] - Convergence tolerance
     * @param {number} [params.C=1.0] - Inverse regularization strength
     */
    constructor({ learningRate = 0.01, maxIter = 1000, tol = 1e-4, C = 1.0 } = {}) {
        this.learningRate = learningRate;
        this.maxIter = maxIter;
        this.tol = tol;
        this.C = C;
        this.weights = null;
        this.classes = null;
        this.nFeatures = null;
    }

    /** @private */
    _sigmoid(z) {
        return z.map(v => 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, v)))));
    }

    /**
     * Fit a binary logistic regression for one class vs rest.
     * @private
     * @param {number[][]} X - Feature matrix
     * @param {number[]} yBin - Binary labels (0 or 1)
     * @returns {number[]} Learned weight vector including bias
     */
    _fitBinary(X, yBin) {
        const n = X.length;
        const d = X[0].length;
        const w = new Array(d + 1).fill(0);

        for (let iter = 0; iter < this.maxIter; iter++) {
            const z = X.map((row, i) =>
                row.reduce((sum, xj, j) => sum + xj * w[j], 0) + w[d]
            );
            const pred = this._sigmoid(z);

            const grad = new Array(d + 1).fill(0);
            for (let i = 0; i < n; i++) {
                const err = pred[i] - yBin[i];
                for (let j = 0; j < d; j++) {
                    grad[j] += err * X[i][j];
                }
                grad[d] += err;
            }

            let maxGrad = 0;
            for (let j = 0; j <= d; j++) {
                const reg = j < d ? w[j] / this.C : 0;
                grad[j] = grad[j] / n + reg;
                maxGrad = Math.max(maxGrad, Math.abs(grad[j]));
            }

            if (maxGrad < this.tol) {
                break;
            }

            const updated = w.map((wj, j) => wj - this.learningRate * grad[j]);
            w.splice(0, w.length, ...updated);
        }

        return w;
    }

    /**
     * Fit the model to training data.
     * @param {number[][]} X - Feature matrix (n_samples x n_features)
     * @param {number[]} y - Class labels
     * @returns {LogisticRegression} this
     */
    fit(X, y) {
        if (!X || !X.length || !y || !y.length) {
            throw new Error('X and y must be non-empty arrays');
        }

        this.classes = [...new Set(y)].sort((a, b) => a - b);
        this.nFeatures = X[0].length;

        if (this.classes.length < 2) {
            throw new Error('At least 2 classes are required');
        }

        if (this.classes.length === 2) {
            const positiveClass = this.classes[1];
            const yBin = y.map(v => (v === positiveClass ? 1 : 0));
            this.weights = [this._fitBinary(X, yBin)];
        } else {
            this.weights = this.classes.map(cls => {
                const yBin = y.map(v => (v === cls ? 1 : 0));
                return this._fitBinary(X, yBin);
            });
        }

        return this;
    }

    /**
     * Predict class probabilities.
     * @param {number[][]} X - Feature matrix
     * @returns {number[][]} Probabilities for each class per sample
     */
    predictProba(X) {
        if (!this.weights) {
            throw new Error('Model has not been fitted yet');
        }

        if (this.classes.length === 2) {
            const w = this.weights[0];
            const d = w.length - 1;
            return X.map(row => {
                const z = row.reduce((sum, xj, j) => sum + xj * w[j], 0) + w[d];
                const p = 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, z))));
                return [1 - p, p];
            });
        }

        return X.map(row => {
            const scores = this.weights.map(w => {
                const d = w.length - 1;
                return row.reduce((sum, xj, j) => sum + xj * w[j], 0) + w[d];
            });

            const maxScore = Math.max(...scores);
            const exps = scores.map(s => Math.exp(s - maxScore));
            const sumExp = exps.reduce((a, b) => a + b, 0);
            return exps.map(e => e / sumExp);
        });
    }

    /**
     * Predict class labels.
     * @param {number[][]} X - Feature matrix
     * @returns {number[]} Predicted class labels
     */
    predict(X) {
        const proba = this.predictProba(X);
        return proba.map(probs => {
            const maxIdx = probs.indexOf(Math.max(...probs));
            return this.classes[maxIdx];
        });
    }

    /**
     * Get model parameters.
     * @returns {Object}
     */
    getParams() {
        return {
            learningRate: this.learningRate,
            maxIter: this.maxIter,
            tol: this.tol,
            C: this.C
        };
    }

    /**
     * Get feature importances based on absolute weight magnitudes.
     * @returns {number[]|null}
     */
    getFeatureImportance() {
        if (!this.weights) {
            return null;
        }

        const d = this.nFeatures;
        const importance = new Array(d).fill(0);

        for (const w of this.weights) {
            for (let j = 0; j < d; j++) {
                importance[j] += Math.abs(w[j]);
            }
        }

        const total = importance.reduce((a, b) => a + b, 0);
        if (total === 0) {
            return importance;
        }
        return importance.map(v => v / total);
    }
}
