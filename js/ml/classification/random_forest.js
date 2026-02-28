/**
 * Random Forest classifier using bootstrap aggregation
 * with random feature subsets.
 * @module classification/random_forest
 */

import { DecisionTreeClassifier } from './decision_tree.js';

/**
 * @class RandomForestClassifier
 */
export class RandomForestClassifier {
    /**
     * @param {Object} params
     * @param {number} [params.nEstimators=100] - Number of trees
     * @param {number} [params.maxDepth=5] - Maximum depth per tree
     * @param {number} [params.minSamplesSplit=2] - Minimum samples to split
     * @param {string|number} [params.maxFeatures='sqrt'] - Features per split: 'sqrt', 'log2', or integer
     */
    constructor({ nEstimators = 100, maxDepth = 5, minSamplesSplit = 2, maxFeatures = 'sqrt' } = {}) {
        this.nEstimators = nEstimators;
        this.maxDepth = maxDepth;
        this.minSamplesSplit = minSamplesSplit;
        this.maxFeatures = maxFeatures;
        this.trees = [];
        this.featureSubsets = [];
        this.classes = null;
        this.nFeatures = null;
    }

    /**
     * Determine the number of features to sample per tree.
     * @private
     * @param {number} totalFeatures
     * @returns {number}
     */
    _getMaxFeatureCount(totalFeatures) {
        if (typeof this.maxFeatures === 'number') {
            return Math.min(this.maxFeatures, totalFeatures);
        }
        if (this.maxFeatures === 'log2') {
            return Math.max(1, Math.floor(Math.log2(totalFeatures)));
        }
        return Math.max(1, Math.floor(Math.sqrt(totalFeatures)));
    }

    /**
     * Generate a bootstrap sample with random feature subset.
     * @private
     * @param {number[][]} X
     * @param {number[]} y
     * @param {number} nFeatSample
     * @returns {{ X: number[][], y: number[], featureIdx: number[] }}
     */
    _bootstrapSample(X, y, nFeatSample) {
        const n = X.length;
        const d = X[0].length;

        const sampleIdx = Array.from({ length: n }, () => Math.floor(Math.random() * n));

        const allFeatures = Array.from({ length: d }, (_, i) => i);
        const shuffled = allFeatures.sort(() => Math.random() - 0.5);
        const featureIdx = shuffled.slice(0, nFeatSample).sort((a, b) => a - b);

        const Xb = sampleIdx.map(i => featureIdx.map(f => X[i][f]));
        const yb = sampleIdx.map(i => y[i]);

        return { X: Xb, y: yb, featureIdx };
    }

    /**
     * Fit the forest to training data.
     * @param {number[][]} X - Feature matrix
     * @param {number[]} y - Class labels
     * @returns {RandomForestClassifier} this
     */
    fit(X, y) {
        if (!X || !X.length || !y || !y.length) {
            throw new Error('X and y must be non-empty arrays');
        }

        this.classes = [...new Set(y)].sort((a, b) => a - b);
        this.nFeatures = X[0].length;
        this.trees = [];
        this.featureSubsets = [];

        const nFeatSample = this._getMaxFeatureCount(this.nFeatures);

        for (let i = 0; i < this.nEstimators; i++) {
            const sample = this._bootstrapSample(X, y, nFeatSample);

            const tree = new DecisionTreeClassifier({
                maxDepth: this.maxDepth,
                minSamplesSplit: this.minSamplesSplit
            });
            tree.fit(sample.X, sample.y);

            this.trees.push(tree);
            this.featureSubsets.push(sample.featureIdx);
        }

        return this;
    }

    /**
     * Project a row to the feature subset used by a specific tree.
     * @private
     * @param {number[]} row
     * @param {number} treeIdx
     * @returns {number[]}
     */
    _projectRow(row, treeIdx) {
        return this.featureSubsets[treeIdx].map(f => row[f]);
    }

    /**
     * Predict class probabilities by averaging tree probabilities.
     * @param {number[][]} X - Feature matrix
     * @returns {number[][]} Averaged probabilities
     */
    predictProba(X) {
        if (this.trees.length === 0) {
            throw new Error('Model has not been fitted yet');
        }

        const nClasses = this.classes.length;

        return X.map(row => {
            const avgProba = new Array(nClasses).fill(0);

            for (let t = 0; t < this.trees.length; t++) {
                const projected = [this._projectRow(row, t)];
                const proba = this.trees[t].predictProba(projected)[0];
                for (let c = 0; c < nClasses; c++) {
                    avgProba[c] += proba[c];
                }
            }

            return avgProba.map(p => p / this.trees.length);
        });
    }

    /**
     * Predict class labels by majority vote.
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
            maxDepth: this.maxDepth,
            minSamplesSplit: this.minSamplesSplit,
            maxFeatures: this.maxFeatures
        };
    }

    /**
     * Aggregate feature importances across all trees.
     * @returns {number[]|null}
     */
    getFeatureImportance() {
        if (this.trees.length === 0) {
            return null;
        }

        const importance = new Array(this.nFeatures).fill(0);

        for (let t = 0; t < this.trees.length; t++) {
            const treeImportance = this.trees[t].getFeatureImportance();
            if (!treeImportance) continue;

            const featureIdx = this.featureSubsets[t];
            for (let j = 0; j < featureIdx.length; j++) {
                importance[featureIdx[j]] += treeImportance[j] || 0;
            }
        }

        const total = importance.reduce((a, b) => a + b, 0);
        if (total === 0) {
            return importance;
        }
        return importance.map(v => v / total);
    }
}
