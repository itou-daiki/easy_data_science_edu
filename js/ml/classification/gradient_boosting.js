/**
 * Gradient Boosting classifier.
 * Binary: fits regression trees to log-odds residuals.
 * Multi-class: one-vs-rest approach.
 * @module classification/gradient_boosting
 */

import { DecisionTreeRegressor } from '../regression/decision_tree.js';

/**
 * @class GradientBoostingClassifier
 */
export class GradientBoostingClassifier {
    /**
     * @param {Object} params
     * @param {number} [params.nEstimators=100] - Number of boosting stages
     * @param {number} [params.learningRate=0.1] - Shrinkage factor
     * @param {number} [params.maxDepth=3] - Maximum depth of each regression tree
     * @param {number} [params.subsample=1.0] - Fraction of samples per stage
     */
    constructor({ nEstimators = 100, learningRate = 0.1, maxDepth = 3, subsample = 1.0 } = {}) {
        this.nEstimators = nEstimators;
        this.learningRate = learningRate;
        this.maxDepth = maxDepth;
        this.subsample = Math.max(0.1, Math.min(1.0, subsample));
        this.models = null;
        this.initialPredictions = null;
        this.classes = null;
        this.nFeatures = null;
    }

    /** @private */
    _sigmoid(z) {
        const clamped = Math.max(-500, Math.min(500, z));
        return 1 / (1 + Math.exp(-clamped));
    }

    /**
     * Subsample indices from [0..n).
     * @private
     * @param {number} n
     * @returns {number[]}
     */
    _sampleIndices(n) {
        const size = Math.max(1, Math.floor(n * this.subsample));
        if (size >= n) {
            return Array.from({ length: n }, (_, i) => i);
        }
        const indices = Array.from({ length: n }, (_, i) => i);
        for (let i = indices.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [indices[i], indices[j]] = [indices[j], indices[i]];
        }
        return indices.slice(0, size);
    }

    /**
     * Fit a binary gradient boosting sequence.
     * @private
     * @param {number[][]} X
     * @param {number[]} yBin - Binary labels (0 or 1)
     * @returns {{ trees: Object[], initPred: number }}
     */
    _fitBinary(X, yBin) {
        const n = X.length;
        const posCount = yBin.filter(v => v === 1).length;
        const negCount = n - posCount;
        const initPred = Math.log((posCount + 1e-7) / (negCount + 1e-7));

        const F = new Array(n).fill(initPred);
        const trees = [];

        for (let m = 0; m < this.nEstimators; m++) {
            const residuals = new Array(n);
            for (let i = 0; i < n; i++) {
                const p = this._sigmoid(F[i]);
                residuals[i] = yBin[i] - p;
            }

            const sampleIdx = this._sampleIndices(n);
            const Xs = sampleIdx.map(i => X[i]);
            const rs = sampleIdx.map(i => residuals[i]);

            const tree = new DecisionTreeRegressor({
                maxDepth: this.maxDepth,
                minSamplesSplit: 2
            });
            tree.fit(Xs, rs);
            trees.push(tree);

            const updates = tree.predict(X);
            for (let i = 0; i < n; i++) {
                F[i] += this.learningRate * updates[i];
            }
        }

        return { trees, initPred };
    }

    /**
     * Fit the model to training data.
     * @param {number[][]} X - Feature matrix
     * @param {number[]} y - Class labels
     * @returns {GradientBoostingClassifier} this
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
            const result = this._fitBinary(X, yBin);
            this.models = [result];
            this.initialPredictions = [result.initPred];
        } else {
            this.models = [];
            this.initialPredictions = [];
            for (const cls of this.classes) {
                const yBin = y.map(v => (v === cls ? 1 : 0));
                const result = this._fitBinary(X, yBin);
                this.models.push(result);
                this.initialPredictions.push(result.initPred);
            }
        }

        return this;
    }

    /**
     * Compute raw score for one class model on a dataset.
     * @private
     * @param {number[][]} X
     * @param {number} modelIdx
     * @returns {number[]}
     */
    _rawScores(X, modelIdx) {
        const { trees, initPred } = this.models[modelIdx];
        const n = X.length;
        const scores = new Array(n).fill(initPred);

        for (const tree of trees) {
            const preds = tree.predict(X);
            for (let i = 0; i < n; i++) {
                scores[i] += this.learningRate * preds[i];
            }
        }

        return scores;
    }

    /**
     * Predict class probabilities.
     * @param {number[][]} X - Feature matrix
     * @returns {number[][]} Probabilities per class per sample
     */
    predictProba(X) {
        if (!this.models) {
            throw new Error('Model has not been fitted yet');
        }

        if (this.classes.length === 2) {
            const scores = this._rawScores(X, 0);
            return scores.map(s => {
                const p = this._sigmoid(s);
                return [1 - p, p];
            });
        }

        const allScores = this.models.map((_, idx) => this._rawScores(X, idx));

        return X.map((_, i) => {
            const scores = allScores.map(s => s[i]);
            const maxScore = Math.max(...scores);
            const exps = scores.map(s => Math.exp(s - maxScore));
            const sumExp = exps.reduce((a, b) => a + b, 0);
            return exps.map(e => e / sumExp);
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
            nEstimators: this.nEstimators,
            learningRate: this.learningRate,
            maxDepth: this.maxDepth,
            subsample: this.subsample
        };
    }

    /**
     * Aggregate feature importances from all regression trees.
     * @returns {number[]|null}
     */
    getFeatureImportance() {
        if (!this.models) {
            return null;
        }

        const importance = new Array(this.nFeatures).fill(0);

        for (const model of this.models) {
            for (const tree of model.trees) {
                const treeImp = tree.getFeatureImportance();
                if (!treeImp) continue;
                for (let j = 0; j < this.nFeatures; j++) {
                    importance[j] += treeImp[j] || 0;
                }
            }
        }

        const total = importance.reduce((a, b) => a + b, 0);
        if (total === 0) {
            return importance;
        }
        return importance.map(v => v / total);
    }
}
