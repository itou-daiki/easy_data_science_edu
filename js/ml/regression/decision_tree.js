/**
 * Decision Tree Regressor using the CART algorithm with MSE criterion.
 * Builds a binary tree by greedily selecting the split that minimises MSE.
 * @module regression/decision_tree
 */

/**
 * @typedef {Object} TreeNode
 * @property {number}    [featureIndex] - Feature used for splitting.
 * @property {number}    [threshold]    - Threshold value.
 * @property {TreeNode}  [left]         - Left child (values <= threshold).
 * @property {TreeNode}  [right]        - Right child (values > threshold).
 * @property {number}    [value]        - Leaf prediction (mean of targets).
 */

/**
 * @class DecisionTreeRegressor
 * @description Regression tree using CART with MSE splitting criterion.
 */
export class DecisionTreeRegressor {
    /**
     * @param {Object}  params                    - Configuration.
     * @param {number}  params.maxDepth            - Max tree depth (default 5).
     * @param {number}  params.minSamplesSplit     - Min samples to attempt a split (default 2).
     * @param {number}  params.minSamplesLeaf      - Min samples in a leaf (default 1).
     */
    constructor(params = {}) {
        /** @type {number} */
        this.maxDepth = params.maxDepth ?? 5;
        /** @type {number} */
        this.minSamplesSplit = params.minSamplesSplit ?? 2;
        /** @type {number} */
        this.minSamplesLeaf = params.minSamplesLeaf ?? 1;
        /** @type {TreeNode|null} */
        this.tree = null;
        /** @type {number} */
        this.nFeatures = 0;
        /** @type {number[]|null} Internal raw importance accumulator */
        this._featureImportances = null;
    }

    /**
     * Compute MSE for an array of values.
     * @param {number[]} values
     * @returns {number}
     */
    static _mse(values) {
        const n = values.length;
        if (n === 0) return 0;
        const mean = values.reduce((a, b) => a + b, 0) / n;
        return values.reduce((acc, v) => acc + (v - mean) ** 2, 0) / n;
    }

    /**
     * Fit the decision tree.
     * @param {number[][]} X - 2D array [nSamples, nFeatures].
     * @param {number[]}   y - 1D array [nSamples].
     * @returns {DecisionTreeRegressor}
     */
    fit(X, y) {
        if (!X || !y || X.length === 0 || y.length === 0) {
            throw new Error('Training data must not be empty.');
        }
        if (X.length !== y.length) {
            throw new Error('X and y must have the same number of samples.');
        }

        this.nFeatures = X[0].length;
        this._featureImportances = Array(this.nFeatures).fill(0);

        const indices = Array.from({ length: X.length }, (_, i) => i);
        this.tree = this._buildTree(X, y, indices, 0);

        return this;
    }

    /**
     * Recursively build the tree.
     * @param {number[][]} X
     * @param {number[]}   y
     * @param {number[]}   indices - Sample indices for this node.
     * @param {number}     depth
     * @param {number[]}   [featureSubset] - Optional subset of feature indices to consider.
     * @returns {TreeNode}
     */
    _buildTree(X, y, indices, depth, featureSubset = null) {
        const n = indices.length;
        const targets = indices.map(i => y[i]);
        const mean = targets.reduce((a, b) => a + b, 0) / n;

        // Stopping conditions
        if (
            depth >= this.maxDepth ||
            n < this.minSamplesSplit ||
            n <= this.minSamplesLeaf
        ) {
            return { value: mean };
        }

        // Check if all targets are identical
        const allSame = targets.every(v => v === targets[0]);
        if (allSame) {
            return { value: mean };
        }

        const featureIndices = featureSubset ?? Array.from({ length: this.nFeatures }, (_, i) => i);
        let bestFeature = -1;
        let bestThreshold = 0;
        let bestScore = Infinity;
        let bestLeftIdx = [];
        let bestRightIdx = [];

        for (const fIdx of featureIndices) {
            // Get unique sorted values for this feature
            const uniqueValues = [...new Set(indices.map(i => X[i][fIdx]))].sort((a, b) => a - b);

            for (let t = 0; t < uniqueValues.length - 1; t++) {
                const threshold = (uniqueValues[t] + uniqueValues[t + 1]) / 2;
                const leftIdx = [];
                const rightIdx = [];

                for (const i of indices) {
                    if (X[i][fIdx] <= threshold) {
                        leftIdx.push(i);
                    } else {
                        rightIdx.push(i);
                    }
                }

                if (leftIdx.length < this.minSamplesLeaf || rightIdx.length < this.minSamplesLeaf) {
                    continue;
                }

                const leftTargets = leftIdx.map(i => y[i]);
                const rightTargets = rightIdx.map(i => y[i]);
                const score =
                    (leftIdx.length * DecisionTreeRegressor._mse(leftTargets) +
                     rightIdx.length * DecisionTreeRegressor._mse(rightTargets)) / n;

                if (score < bestScore) {
                    bestScore = score;
                    bestFeature = fIdx;
                    bestThreshold = threshold;
                    bestLeftIdx = leftIdx;
                    bestRightIdx = rightIdx;
                }
            }
        }

        // No valid split found
        if (bestFeature === -1) {
            return { value: mean };
        }

        // Accumulate feature importance (weighted MSE reduction)
        const parentMSE = DecisionTreeRegressor._mse(targets);
        const reduction = n * parentMSE - bestScore * n;
        if (this._featureImportances) {
            this._featureImportances[bestFeature] += reduction;
        }

        const leftChild = this._buildTree(X, y, bestLeftIdx, depth + 1, featureSubset);
        const rightChild = this._buildTree(X, y, bestRightIdx, depth + 1, featureSubset);

        return {
            featureIndex: bestFeature,
            threshold: bestThreshold,
            left: leftChild,
            right: rightChild,
        };
    }

    /**
     * Traverse the tree for a single sample.
     * @param {TreeNode}  node
     * @param {number[]}  sample
     * @returns {number}
     */
    _predictSample(node, sample) {
        if (node.value !== undefined && !node.left && !node.right) {
            return node.value;
        }
        if (sample[node.featureIndex] <= node.threshold) {
            return this._predictSample(node.left, sample);
        }
        return this._predictSample(node.right, sample);
    }

    /**
     * Predict target values.
     * @param {number[][]} X - 2D array [nSamples, nFeatures].
     * @returns {number[]}
     */
    predict(X) {
        if (!this.tree) {
            throw new Error('Model has not been fitted yet. Call fit() first.');
        }
        if (!X || X.length === 0) {
            return [];
        }
        return X.map(row => this._predictSample(this.tree, row));
    }

    /**
     * Return model parameters.
     * @returns {{ maxDepth: number, minSamplesSplit: number, minSamplesLeaf: number }}
     */
    getParams() {
        return {
            maxDepth: this.maxDepth,
            minSamplesSplit: this.minSamplesSplit,
            minSamplesLeaf: this.minSamplesLeaf,
        };
    }

    /**
     * Return normalised feature importances (total MSE reduction per feature).
     * @returns {number[]|null}
     */
    getFeatureImportance() {
        if (!this._featureImportances) {
            return null;
        }
        const total = this._featureImportances.reduce((a, b) => a + b, 0);
        if (total === 0) {
            return this._featureImportances.map(() => 0);
        }
        return this._featureImportances.map(v => v / total);
    }
}
