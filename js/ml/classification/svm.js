/**
 * Linear Support Vector Machine classifier.
 * Uses hinge loss with SGD optimization.
 * Multi-class via one-vs-rest strategy.
 * @module classification/svm
 */

/**
 * @class SVMClassifier
 */
export class SVMClassifier {
    /**
     * @param {Object} params
     * @param {number} [params.C=1.0] - Regularization parameter
     * @param {number} [params.learningRate=0.001] - SGD learning rate
     * @param {number} [params.maxIter=1000] - Maximum training iterations
     */
    constructor({ C = 1.0, learningRate = 0.001, maxIter = 1000 } = {}) {
        this.C = C;
        this.learningRate = learningRate;
        this.maxIter = maxIter;
        this.weights = null;
        this.classes = null;
        this.nFeatures = null;
    }

    /**
     * Train a binary SVM using SGD with hinge loss.
     * @private
     * @param {number[][]} X - Feature matrix
     * @param {number[]} yBin - Labels in {-1, +1}
     * @returns {number[]} Weight vector including bias at last position
     */
    _fitBinary(X, yBin) {
        const n = X.length;
        const d = X[0].length;
        const w = new Array(d + 1).fill(0);

        for (let iter = 0; iter < this.maxIter; iter++) {
            const idx = Math.floor(Math.random() * n);
            const xi = X[idx];
            const yi = yBin[idx];

            const decision = xi.reduce((sum, xj, j) => sum + xj * w[j], 0) + w[d];
            const lr = this.learningRate / (1 + iter * 0.0001);

            if (yi * decision < 1) {
                for (let j = 0; j < d; j++) {
                    w[j] = w[j] - lr * (w[j] / this.C - yi * xi[j]);
                }
                w[d] = w[d] + lr * yi;
            } else {
                for (let j = 0; j < d; j++) {
                    w[j] = w[j] - lr * (w[j] / this.C);
                }
            }
        }

        return w;
    }

    /**
     * Compute raw decision values for a binary model.
     * @private
     * @param {number[][]} X
     * @param {number[]} w - Weights including bias
     * @returns {number[]}
     */
    _decisionFunction(X, w) {
        const d = w.length - 1;
        return X.map(row =>
            row.reduce((sum, xj, j) => sum + xj * w[j], 0) + w[d]
        );
    }

    /**
     * Convert decision value to probability via sigmoid (Platt scaling approximation).
     * @private
     * @param {number} decision
     * @returns {number}
     */
    _sigmoid(decision) {
        const z = Math.max(-500, Math.min(500, decision));
        return 1 / (1 + Math.exp(-z));
    }

    /**
     * Fit the model to training data.
     * @param {number[][]} X - Feature matrix
     * @param {number[]} y - Class labels
     * @returns {SVMClassifier} this
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
            const yBin = y.map(v => (v === positiveClass ? 1 : -1));
            this.weights = [this._fitBinary(X, yBin)];
        } else {
            this.weights = this.classes.map(cls => {
                const yBin = y.map(v => (v === cls ? 1 : -1));
                return this._fitBinary(X, yBin);
            });
        }

        return this;
    }

    /**
     * Predict class probabilities using sigmoid approximation of decision values.
     * @param {number[][]} X - Feature matrix
     * @returns {number[][]} Probabilities per class per sample
     */
    predictProba(X) {
        if (!this.weights) {
            throw new Error('Model has not been fitted yet');
        }

        if (this.classes.length === 2) {
            const decisions = this._decisionFunction(X, this.weights[0]);
            return decisions.map(d => {
                const p = this._sigmoid(d);
                return [1 - p, p];
            });
        }

        return X.map(row => {
            const scores = this.weights.map(w => {
                const d = w.length - 1;
                return row.reduce((sum, xj, j) => sum + xj * w[j], 0) + w[d];
            });

            const probs = scores.map(s => this._sigmoid(s));
            const total = probs.reduce((a, b) => a + b, 0);
            return total > 0
                ? probs.map(p => p / total)
                : probs.map(() => 1 / this.classes.length);
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
            C: this.C,
            learningRate: this.learningRate,
            maxIter: this.maxIter
        };
    }

    /**
     * Feature importances based on absolute weight magnitudes.
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
