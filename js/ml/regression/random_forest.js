/**
 * Random Forest Regressor.
 * Ensemble of DecisionTreeRegressors trained on bootstrap samples
 * with random feature subsets. Predictions are averaged across all trees.
 * @module regression/random_forest
 */

import { DecisionTreeRegressor } from './decision_tree.js';
import { createSeededRandom, randomInt } from '../random.js';

/**
 * @class RandomForestRegressor
 * @description Bagging ensemble of regression trees with feature subsampling.
 */
export class RandomForestRegressor {
    /**
     * @param {Object}         params                   - Configuration.
     * @param {number}         params.nEstimators        - Number of trees (default 100).
     * @param {number}         params.maxDepth           - Max depth per tree (default 5).
     * @param {number}         params.minSamplesSplit    - Min samples to split (default 2).
     * @param {string|number}  params.maxFeatures        - Features per split: 'sqrt', 'log2', number, or null for all (default 'sqrt').
     * @param {number}         params.randomState        - Seed for bootstrap and feature sampling (default 42).
     */
    constructor(params = {}) {
        /** @type {number} */
        this.nEstimators = params.nEstimators ?? 100;
        /** @type {number} */
        this.maxDepth = params.maxDepth ?? 5;
        /** @type {number} */
        this.minSamplesSplit = params.minSamplesSplit ?? 2;
        /** @type {string|number} */
        this.maxFeatures = 'maxFeatures' in params ? params.maxFeatures : 'sqrt';
        /** @type {number} */
        this.randomState = params.randomState ?? 42;
        /** @type {DecisionTreeRegressor[]} */
        this.trees = [];
        /** @type {number} */
        this.nFeatures = 0;
    }

    /**
     * Draw a bootstrap sample (sampling with replacement).
     * @param {number} n - Population size.
     * @returns {number[]} Array of sampled indices.
     */
    static _bootstrapIndices(n, rng) {
        const indices = [];
        for (let i = 0; i < n; i++) {
            indices.push(randomInt(rng, n));
        }
        return indices;
    }

    /**
     * Fit the random forest.
     * @param {number[][]} X - 2D array [nSamples, nFeatures].
     * @param {number[]}   y - 1D array [nSamples].
     * @returns {RandomForestRegressor}
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
        const rng = createSeededRandom(this.randomState);

        this.trees = [];

        for (let t = 0; t < this.nEstimators; t++) {
            // Bootstrap sample
            const sampleIdx = RandomForestRegressor._bootstrapIndices(n, rng);
            const Xb = sampleIdx.map(i => X[i]);
            const yb = sampleIdx.map(i => y[i]);

            const tree = new DecisionTreeRegressor({
                maxDepth: this.maxDepth,
                minSamplesSplit: this.minSamplesSplit,
                maxFeatures: this.maxFeatures,
                randomState: randomInt(rng, 0x7fffffff),
            });
            tree.fit(Xb, yb);
            this.trees.push(tree);
        }

        return this;
    }

    /**
     * Predict by averaging predictions across all trees.
     * @param {number[][]} X - 2D array [nSamples, nFeatures].
     * @returns {number[]}
     */
    predict(X) {
        if (this.trees.length === 0) {
            throw new Error('Model has not been fitted yet. Call fit() first.');
        }
        if (!X || X.length === 0) {
            return [];
        }

        const nSamples = X.length;
        const sums = Array(nSamples).fill(0);

        for (let t = 0; t < this.trees.length; t++) {
            const preds = this.trees[t].predict(X);
            for (let i = 0; i < nSamples; i++) {
                sums[i] += preds[i];
            }
        }

        return sums.map(s => s / this.trees.length);
    }

    /**
     * Return model parameters.
     * @returns {{ nEstimators: number, maxDepth: number, minSamplesSplit: number, maxFeatures: string|number, randomState: number }}
     */
    getParams() {
        return {
            nEstimators: this.nEstimators,
            maxDepth: this.maxDepth,
            minSamplesSplit: this.minSamplesSplit,
            maxFeatures: this.maxFeatures,
            randomState: this.randomState,
        };
    }

    /**
     * Average feature importances across all trees, mapped back to original feature indices.
     * @returns {number[]|null}
     */
    getFeatureImportance() {
        if (this.trees.length === 0) {
            return null;
        }

        const importances = Array(this.nFeatures).fill(0);

        for (let t = 0; t < this.trees.length; t++) {
            const treeImp = this.trees[t].getFeatureImportance();
            if (!treeImp) continue;
            for (let j = 0; j < this.nFeatures; j++) {
                importances[j] += treeImp[j] || 0;
            }
        }

        const total = importances.reduce((a, b) => a + b, 0);
        if (total === 0) {
            return importances.map(() => 0);
        }
        return importances.map(v => v / total);
    }
}
