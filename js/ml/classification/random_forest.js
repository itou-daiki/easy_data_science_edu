/**
 * Random Forest classifier using bootstrap aggregation
 * with random feature subsets.
 * @module classification/random_forest
 */

import { DecisionTreeClassifier } from './decision_tree.js';
import { createSeededRandom, randomInt } from '../random.js';

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
     * @param {number} [params.randomState=42] - Seed for bootstrap and feature sampling
     */
    constructor({ nEstimators = 100, maxDepth = 5, minSamplesSplit = 2, maxFeatures = 'sqrt', randomState = 42 } = {}) {
        this.nEstimators = nEstimators;
        this.maxDepth = maxDepth;
        this.minSamplesSplit = minSamplesSplit;
        this.maxFeatures = maxFeatures;
        this.randomState = randomState;
        this.trees = [];
        this.classes = null;
        this.nFeatures = null;
    }

    /**
     * Generate a bootstrap sample.
     * @private
     * @param {number[][]} X
     * @param {number[]} y
     * @param {() => number} rng
     * @returns {{ X: number[][], y: number[] }}
     */
    _bootstrapSample(X, y, rng) {
        const n = X.length;
        const sampleIdx = Array.from({ length: n }, () => randomInt(rng, n));
        const Xb = sampleIdx.map(i => [...X[i]]);
        const yb = sampleIdx.map(i => y[i]);
        return { X: Xb, y: yb };
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
        const rng = createSeededRandom(this.randomState);

        for (let i = 0; i < this.nEstimators; i++) {
            const sample = this._bootstrapSample(X, y, rng);

            const tree = new DecisionTreeClassifier({
                maxDepth: this.maxDepth,
                minSamplesSplit: this.minSamplesSplit,
                maxFeatures: this.maxFeatures,
                randomState: randomInt(rng, 0x7fffffff),
                classes: this.classes
            });
            tree.fit(sample.X, sample.y);

            this.trees.push(tree);
        }

        return this;
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
                const tree = this.trees[t];
                const proba = tree.predictProba([row])[0];
                for (let localIndex = 0; localIndex < tree.classes.length; localIndex++) {
                    const globalIndex = this.classes.indexOf(tree.classes[localIndex]);
                    if (globalIndex >= 0) avgProba[globalIndex] += proba[localIndex] || 0;
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
            maxFeatures: this.maxFeatures,
            randomState: this.randomState
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

            for (let j = 0; j < this.nFeatures; j++) {
                importance[j] += treeImportance[j] || 0;
            }
        }

        const total = importance.reduce((a, b) => a + b, 0);
        if (total === 0) {
            return importance;
        }
        return importance.map(v => v / total);
    }
}
