/**
 * Gaussian Naive Bayes classifier.
 * Assumes features follow a normal distribution per class.
 * @module classification/naive_bayes
 */

/**
 * @class GaussianNaiveBayes
 */
export class GaussianNaiveBayes {
    constructor() {
        this.classes = null;
        this.classPriors = null;
        this.classMeans = null;
        this.classVars = null;
        this.nFeatures = null;
    }

    /**
     * Compute mean of an array.
     * @private
     * @param {number[]} arr
     * @returns {number}
     */
    _mean(arr) {
        return arr.reduce((a, b) => a + b, 0) / arr.length;
    }

    /**
     * Compute variance of an array.
     * @private
     * @param {number[]} arr
     * @param {number} mean
     * @returns {number}
     */
    _variance(arr, mean) {
        const sumSq = arr.reduce((s, v) => s + (v - mean) ** 2, 0);
        return sumSq / arr.length;
    }

    /**
     * Gaussian log-probability density.
     * @private
     * @param {number} x - Value
     * @param {number} mean - Distribution mean
     * @param {number} variance - Distribution variance
     * @returns {number} Log probability density
     */
    _gaussianLogPdf(x, mean, variance) {
        const safeVar = Math.max(variance, 1e-9);
        return -0.5 * Math.log(2 * Math.PI * safeVar) - ((x - mean) ** 2) / (2 * safeVar);
    }

    /**
     * Fit the model by computing per-class statistics.
     * @param {number[][]} X - Feature matrix
     * @param {number[]} y - Class labels
     * @returns {GaussianNaiveBayes} this
     */
    fit(X, y) {
        if (!X || !X.length || !y || !y.length) {
            throw new Error('X and y must be non-empty arrays');
        }

        this.classes = [...new Set(y)].sort((a, b) => a - b);
        this.nFeatures = X[0].length;
        const n = y.length;

        this.classPriors = {};
        this.classMeans = {};
        this.classVars = {};

        for (const cls of this.classes) {
            const indices = [];
            for (let i = 0; i < n; i++) {
                if (y[i] === cls) indices.push(i);
            }

            this.classPriors[cls] = indices.length / n;

            const means = new Array(this.nFeatures);
            const vars = new Array(this.nFeatures);

            for (let f = 0; f < this.nFeatures; f++) {
                const values = indices.map(i => X[i][f]);
                const m = this._mean(values);
                means[f] = m;
                vars[f] = this._variance(values, m);
            }

            this.classMeans[cls] = means;
            this.classVars[cls] = vars;
        }

        return this;
    }

    /**
     * Predict class probabilities using Bayes theorem.
     * @param {number[][]} X - Feature matrix
     * @returns {number[][]} Probabilities for each class per sample
     */
    predictProba(X) {
        if (!this.classPriors) {
            throw new Error('Model has not been fitted yet');
        }

        return X.map(row => {
            const logProbs = this.classes.map(cls => {
                let logP = Math.log(this.classPriors[cls]);
                for (let f = 0; f < this.nFeatures; f++) {
                    logP += this._gaussianLogPdf(
                        row[f],
                        this.classMeans[cls][f],
                        this.classVars[cls][f]
                    );
                }
                return logP;
            });

            const maxLog = Math.max(...logProbs);
            const exps = logProbs.map(lp => Math.exp(lp - maxLog));
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
        return {};
    }

    /**
     * Feature importance based on variance ratio between classes.
     * Higher ratio means the feature discriminates classes better.
     * @returns {number[]|null}
     */
    getFeatureImportance() {
        if (!this.classMeans) {
            return null;
        }

        const importance = new Array(this.nFeatures).fill(0);
        const overallMeans = new Array(this.nFeatures).fill(0);

        for (const cls of this.classes) {
            const prior = this.classPriors[cls];
            for (let f = 0; f < this.nFeatures; f++) {
                overallMeans[f] += prior * this.classMeans[cls][f];
            }
        }

        for (let f = 0; f < this.nFeatures; f++) {
            let betweenVar = 0;
            let withinVar = 0;
            for (const cls of this.classes) {
                const prior = this.classPriors[cls];
                betweenVar += prior * (this.classMeans[cls][f] - overallMeans[f]) ** 2;
                withinVar += prior * this.classVars[cls][f];
            }
            importance[f] = withinVar > 1e-12 ? betweenVar / withinVar : 0;
        }

        const total = importance.reduce((a, b) => a + b, 0);
        if (total === 0) {
            return importance;
        }
        return importance.map(v => v / total);
    }
}
