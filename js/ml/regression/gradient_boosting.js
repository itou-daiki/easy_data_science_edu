/**
 * Gradient Boosting Regressor.
 * Builds an additive model of shallow decision trees, each fitted to the
 * negative gradient (residuals) of the MSE loss.
 * @module regression/gradient_boosting
 */

import { DecisionTreeRegressor } from './decision_tree.js';

/**
 * @class GradientBoostingRegressor
 * @description Sequential ensemble of regression trees fitting residuals.
 */
export class GradientBoostingRegressor {
    /**
     * @param {Object}  params                - Configuration.
     * @param {number}  params.nEstimators     - Number of boosting stages (default 100).
     * @param {number}  params.learningRate    - Shrinkage factor (default 0.1).
     * @param {number}  params.maxDepth        - Max depth per tree (default 3).
     * @param {number}  params.subsample       - Fraction of samples per tree (default 1.0).
     */
    constructor(params = {}) {
        /** @type {number} */
        this.nEstimators = params.nEstimators ?? 100;
        /** @type {number} */
        this.learningRate = params.learningRate ?? 0.1;
        /** @type {number} */
        this.maxDepth = params.maxDepth ?? 3;
        /** @type {number} */
        this.subsample = params.subsample ?? 1.0;
        /** @type {DecisionTreeRegressor[]} */
        this.trees = [];
        /** @type {number|null} Initial prediction (mean of training targets) */
        this.initialPrediction = null;
        /** @type {number} */
        this.nFeatures = 0;
    }

    /**
     * Sample a subset of indices without replacement.
     * @param {number} n         - Total population size.
     * @param {number} fraction  - Fraction to sample (0, 1].
     * @returns {number[]}
     */
    static _subsampleIndices(n, fraction) {
        if (fraction >= 1.0) {
            return Array.from({ length: n }, (_, i) => i);
        }
        const k = Math.max(1, Math.floor(n * fraction));
        const pool = Array.from({ length: n }, (_, i) => i);
        // Fisher-Yates partial shuffle
        for (let i = pool.length - 1; i > pool.length - 1 - k && i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [pool[i], pool[j]] = [pool[j], pool[i]];
        }
        return pool.slice(pool.length - k);
    }

    /**
     * Fit the gradient boosting model.
     * @param {number[][]} X - 2D array [nSamples, nFeatures].
     * @param {number[]}   y - 1D array [nSamples].
     * @returns {GradientBoostingRegressor}
     */
    fit(X, y) {
        if (!X || !y || X.length === 0 || y.length === 0) {
            throw new Error('Training data must not be empty.');
        }
        if (X.length !== y.length) {
            throw new Error('X and y must have the same number of samples.');
        }

        const n = X.length;
        this.nFeatures = X[0].length;

        // F_0 = mean(y)
        this.initialPrediction = y.reduce((a, b) => a + b, 0) / n;

        // Current predictions for each sample
        let predictions = Array(n).fill(this.initialPrediction);
        this.trees = [];

        for (let stage = 0; stage < this.nEstimators; stage++) {
            // Compute residuals (negative gradient of MSE loss = y - F)
            const residuals = y.map((yi, i) => yi - predictions[i]);

            // Subsample
            const sampleIdx = GradientBoostingRegressor._subsampleIndices(n, this.subsample);
            const Xsub = sampleIdx.map(i => X[i]);
            const rSub = sampleIdx.map(i => residuals[i]);

            // Fit a shallow tree to residuals
            const tree = new DecisionTreeRegressor({
                maxDepth: this.maxDepth,
                minSamplesSplit: 2,
                minSamplesLeaf: 1,
            });
            tree.fit(Xsub, rSub);
            this.trees.push(tree);

            // Update predictions: F_{m+1} = F_m + lr * h_m(x)
            const treePreds = tree.predict(X);
            predictions = predictions.map((p, i) => p + this.learningRate * treePreds[i]);
        }

        return this;
    }

    /**
     * Predict target values.
     * @param {number[][]} X - 2D array [nSamples, nFeatures].
     * @returns {number[]}
     */
    predict(X) {
        if (this.trees.length === 0 || this.initialPrediction === null) {
            throw new Error('Model has not been fitted yet. Call fit() first.');
        }
        if (!X || X.length === 0) {
            return [];
        }

        const nSamples = X.length;
        const predictions = Array(nSamples).fill(this.initialPrediction);

        for (const tree of this.trees) {
            const treePreds = tree.predict(X);
            for (let i = 0; i < nSamples; i++) {
                predictions[i] += this.learningRate * treePreds[i];
            }
        }

        return predictions;
    }

    /**
     * Return model parameters.
     * @returns {{ nEstimators: number, learningRate: number, maxDepth: number, subsample: number }}
     */
    getParams() {
        return {
            nEstimators: this.nEstimators,
            learningRate: this.learningRate,
            maxDepth: this.maxDepth,
            subsample: this.subsample,
        };
    }

    /**
     * Average feature importances across all boosting stages.
     * @returns {number[]|null}
     */
    getFeatureImportance() {
        if (this.trees.length === 0) {
            return null;
        }

        const importances = Array(this.nFeatures).fill(0);

        for (const tree of this.trees) {
            const treeImp = tree.getFeatureImportance();
            if (!treeImp) continue;
            for (let j = 0; j < treeImp.length; j++) {
                importances[j] += treeImp[j];
            }
        }

        const total = importances.reduce((a, b) => a + b, 0);
        if (total === 0) {
            return importances.map(() => 0);
        }
        return importances.map(v => v / total);
    }
}
