/**
 * Decision Tree classifier using CART algorithm.
 * Supports Gini impurity and entropy criteria.
 * @module classification/decision_tree
 */

/**
 * @class DecisionTreeClassifier
 */
export class DecisionTreeClassifier {
    /**
     * @param {Object} params
     * @param {number} [params.maxDepth=5] - Maximum tree depth
     * @param {number} [params.minSamplesSplit=2] - Minimum samples to split a node
     * @param {number} [params.minSamplesLeaf=1] - Minimum samples in a leaf
     * @param {string} [params.criterion='gini'] - Split criterion: 'gini' or 'entropy'
     */
    constructor({ maxDepth = 5, minSamplesSplit = 2, minSamplesLeaf = 1, criterion = 'gini' } = {}) {
        this.maxDepth = maxDepth;
        this.minSamplesSplit = minSamplesSplit;
        this.minSamplesLeaf = minSamplesLeaf;
        this.criterion = criterion;
        this.tree = null;
        this.classes = null;
        this.nFeatures = null;
        this.featureImportances = null;
    }

    /**
     * Compute impurity for a set of labels.
     * @private
     * @param {number[]} labels
     * @returns {number}
     */
    _impurity(labels) {
        if (labels.length === 0) return 0;
        const counts = {};
        for (const l of labels) {
            counts[l] = (counts[l] || 0) + 1;
        }
        const n = labels.length;
        const probs = Object.values(counts).map(c => c / n);

        if (this.criterion === 'entropy') {
            return -probs.reduce((s, p) => s + (p > 0 ? p * Math.log2(p) : 0), 0);
        }
        return 1 - probs.reduce((s, p) => s + p * p, 0);
    }

    /**
     * Build the tree recursively.
     * @private
     * @param {number[][]} X
     * @param {number[]} y
     * @param {number} depth
     * @returns {Object} Tree node
     */
    _buildTree(X, y, depth) {
        const classDist = this._classDistribution(y);

        if (depth >= this.maxDepth || y.length < this.minSamplesSplit || new Set(y).size === 1) {
            return { leaf: true, classDist, prediction: this._majorityClass(y) };
        }

        const best = this._findBestSplit(X, y);
        if (best === null) {
            return { leaf: true, classDist, prediction: this._majorityClass(y) };
        }

        const { featureIdx, threshold, leftIdx, rightIdx, gain } = best;
        this.featureImportances[featureIdx] += gain * y.length;

        const leftX = leftIdx.map(i => X[i]);
        const leftY = leftIdx.map(i => y[i]);
        const rightX = rightIdx.map(i => X[i]);
        const rightY = rightIdx.map(i => y[i]);

        return {
            leaf: false,
            featureIdx,
            threshold,
            left: this._buildTree(leftX, leftY, depth + 1),
            right: this._buildTree(rightX, rightY, depth + 1)
        };
    }

    /**
     * Find the best split across all features.
     * @private
     * @param {number[][]} X
     * @param {number[]} y
     * @returns {Object|null}
     */
    _findBestSplit(X, y) {
        const n = y.length;
        const parentImpurity = this._impurity(y);
        let bestGain = 0;
        let bestSplit = null;

        for (let f = 0; f < this.nFeatures; f++) {
            const values = X.map(row => row[f]);
            const unique = [...new Set(values)].sort((a, b) => a - b);

            for (let t = 0; t < unique.length - 1; t++) {
                const threshold = (unique[t] + unique[t + 1]) / 2;
                const leftIdx = [];
                const rightIdx = [];
                for (let i = 0; i < n; i++) {
                    if (X[i][f] <= threshold) {
                        leftIdx.push(i);
                    } else {
                        rightIdx.push(i);
                    }
                }

                if (leftIdx.length < this.minSamplesLeaf || rightIdx.length < this.minSamplesLeaf) {
                    continue;
                }

                const leftY = leftIdx.map(i => y[i]);
                const rightY = rightIdx.map(i => y[i]);
                const gain = parentImpurity
                    - (leftY.length / n) * this._impurity(leftY)
                    - (rightY.length / n) * this._impurity(rightY);

                if (gain > bestGain) {
                    bestGain = gain;
                    bestSplit = { featureIdx: f, threshold, leftIdx, rightIdx, gain };
                }
            }
        }

        return bestSplit;
    }

    /** @private */
    _classDistribution(y) {
        const dist = {};
        for (const cls of this.classes) {
            dist[cls] = 0;
        }
        for (const label of y) {
            dist[label] = (dist[label] || 0) + 1;
        }
        const n = y.length;
        const result = {};
        for (const cls of this.classes) {
            result[cls] = (dist[cls] || 0) / n;
        }
        return result;
    }

    /** @private */
    _majorityClass(y) {
        const counts = {};
        for (const label of y) {
            counts[label] = (counts[label] || 0) + 1;
        }
        let best = null;
        let bestCount = -1;
        for (const [cls, count] of Object.entries(counts)) {
            if (count > bestCount) {
                bestCount = count;
                best = Number(cls);
            }
        }
        return best;
    }

    /** @private */
    _predictOne(row, node) {
        if (node.leaf) {
            return node;
        }
        if (row[node.featureIdx] <= node.threshold) {
            return this._predictOne(row, node.left);
        }
        return this._predictOne(row, node.right);
    }

    /**
     * Fit the model to training data.
     * @param {number[][]} X - Feature matrix
     * @param {number[]} y - Class labels
     * @returns {DecisionTreeClassifier} this
     */
    fit(X, y) {
        if (!X || !X.length || !y || !y.length) {
            throw new Error('X and y must be non-empty arrays');
        }
        this.classes = [...new Set(y)].sort((a, b) => a - b);
        this.nFeatures = X[0].length;
        this.featureImportances = new Array(this.nFeatures).fill(0);
        this.tree = this._buildTree(X, y, 0);

        const total = this.featureImportances.reduce((a, b) => a + b, 0);
        if (total > 0) {
            this.featureImportances = this.featureImportances.map(v => v / total);
        }
        return this;
    }

    /**
     * Predict class probabilities.
     * @param {number[][]} X - Feature matrix
     * @returns {number[][]} Class probabilities per sample
     */
    predictProba(X) {
        if (!this.tree) {
            throw new Error('Model has not been fitted yet');
        }
        return X.map(row => {
            const node = this._predictOne(row, this.tree);
            return this.classes.map(cls => node.classDist[cls] || 0);
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
            maxDepth: this.maxDepth,
            minSamplesSplit: this.minSamplesSplit,
            minSamplesLeaf: this.minSamplesLeaf,
            criterion: this.criterion
        };
    }

    /** @returns {number[]|null} */
    getFeatureImportance() {
        return this.featureImportances ? [...this.featureImportances] : null;
    }
}
