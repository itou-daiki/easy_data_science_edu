/**
 * shap.js - SHAP (SHapley Additive exPlanations) value computation
 *
 * Provides two methods:
 * 1. linearSHAP - Exact SHAP for linear models (O(n*p))
 * 2. kernelSHAP - Seeded permutation approximation for any model
 *
 * @module shap
 */

import { createSeededRandom, shuffleCopy } from './random.js';

// ===========================================================================
// Linear SHAP (exact for linear models)
// ===========================================================================

/**
 * Compute exact SHAP values for linear models.
 *
 * For linear models: SHAP_i = w_i * (x_i - E[x_i])
 * Base value = intercept + sum(w_i * E[x_i]) = E[f(x)]
 *
 * @param {number[]} coefficients - Model weights (one per feature)
 * @param {number} intercept - Model intercept/bias
 * @param {number[][]} X - Instances to explain (n x p)
 * @param {number[]} featureMeans - Mean of each feature from training data
 * @returns {{ shapValues: number[][], baseValue: number }}
 */
export function linearSHAP(coefficients, intercept, X, featureMeans) {
    const baseValue = intercept + coefficients.reduce(
        (sum, w, i) => sum + w * featureMeans[i], 0
    );

    const shapValues = X.map(x =>
        coefficients.map((w, i) => w * (x[i] - featureMeans[i]))
    );

    return { shapValues, baseValue };
}

// ===========================================================================
// Kernel SHAP (model-agnostic)
// ===========================================================================

/**
 * Compute model-agnostic interventional SHAP values with sampled permutations.
 *
 * Each sampled path starts from a background row and adds features in a
 * random order. The path increments telescope exactly to f(x)-f(background),
 * so local additivity is preserved even when only a subset of permutations is
 * sampled. Correlated features can still share credit differently depending on
 * the supplied background distribution.
 *
 * @param {function(number[][]): number[]} predictFn - Prediction function (batch input → batch output)
 * @param {number[][]} X - Instances to explain (n x p)
 * @param {number[][]} background - Background dataset for marginalization
 * @param {Object} [options]
 * @param {number} [options.maxBackground=50] - Max background samples to use
 * @param {number} [options.nPermutations] - Permutations per instance
 * @param {number} [options.maxEvaluations=50000] - Approximate total prediction budget
 * @param {number} [options.randomState=42] - Deterministic seed
 * @returns {{ shapValues: number[][], baseValue: number, nPermutations: number, maxAdditivityError: number }}
 */
export function kernelSHAP(predictFn, X, background, options = {}) {
    const {
        maxBackground = 50,
        maxEvaluations = 50000,
        randomState = 42,
    } = options;
    if (typeof predictFn !== 'function') throw new Error('kernelSHAP: predictFn must be a function');
    if (!Array.isArray(X) || X.length === 0 || !Array.isArray(X[0]) || X[0].length === 0) {
        throw new Error('kernelSHAP: X must be a non-empty 2D array');
    }
    if (!Array.isArray(background) || background.length === 0 || !Array.isArray(background[0])) {
        throw new Error('kernelSHAP: background must be a non-empty 2D array');
    }

    const bg = background.length > maxBackground
        ? background.slice(0, maxBackground)
        : background;
    const nFeatures = X[0].length;
    if (X.some(row => row.length !== nFeatures) || bg.some(row => row.length !== nFeatures)) {
        throw new Error('kernelSHAP: all rows must have the same feature count');
    }

    const bgPreds = predictFn(bg);
    if (!Array.isArray(bgPreds) || bgPreds.length !== bg.length || bgPreds.some(value => !Number.isFinite(value))) {
        throw new Error('kernelSHAP: predictFn returned invalid background predictions');
    }
    const baseValue = bgPreds.reduce((a, b) => a + b, 0) / bgPreds.length;
    const budgetPermutations = Math.max(1, Math.floor(maxEvaluations / Math.max(1, X.length * bg.length * nFeatures)));
    const nPermutations = Math.max(1, Math.min(
        Number.isInteger(options.nPermutations) ? options.nPermutations : 64,
        budgetPermutations
    ));
    const rng = createSeededRandom(randomState);
    let maxAdditivityError = 0;
    const shapValues = X.map(x => {
        const values = _permutationSHAPInstance(predictFn, x, bg, bgPreds, nPermutations, rng);
        const prediction = predictFn([x])[0];
        if (!Number.isFinite(prediction)) throw new Error('kernelSHAP: predictFn returned a non-finite value');
        const explained = baseValue + values.reduce((sum, value) => sum + value, 0);
        const residual = prediction - explained;
        maxAdditivityError = Math.max(maxAdditivityError, Math.abs(residual));
        if (Math.abs(residual) > 1e-12) values[values.length - 1] += residual;
        return values;
    });

    return { shapValues, baseValue, nPermutations, maxAdditivityError };
}

/**
 * Compute SHAP values for a single instance using Kernel SHAP.
 * @private
 */
function _permutationSHAPInstance(predictFn, x, background, backgroundPredictions, nPermutations, rng) {
    const nFeatures = x.length;
    const contributions = Array(nFeatures).fill(0);
    const featureIndices = Array.from({ length: nFeatures }, (_, index) => index);

    for (let iteration = 0; iteration < nPermutations; iteration++) {
        const permutation = shuffleCopy(featureIndices, rng);
        const workingRows = background.map(row => [...row]);
        let previous = [...backgroundPredictions];
        for (const featureIndex of permutation) {
            for (const row of workingRows) row[featureIndex] = x[featureIndex];
            const current = predictFn(workingRows);
            if (!Array.isArray(current) || current.length !== workingRows.length || current.some(value => !Number.isFinite(value))) {
                throw new Error('kernelSHAP: predictFn returned invalid predictions');
            }
            for (let rowIndex = 0; rowIndex < current.length; rowIndex++) {
                contributions[featureIndex] += current[rowIndex] - previous[rowIndex];
            }
            previous = current;
        }
    }

    const denominator = nPermutations * background.length;
    return contributions.map(value => value / denominator);
}

// ===========================================================================
// SHAP Summary Statistics
// ===========================================================================

/**
 * Compute mean absolute SHAP values per feature (for global importance).
 *
 * @param {number[][]} shapValues - SHAP values (n x p)
 * @returns {{ meanAbsSHAP: number[], meanSHAP: number[] }}
 */
export function shapSummary(shapValues) {
    if (shapValues.length === 0) return { meanAbsSHAP: [], meanSHAP: [] };

    const nFeatures = shapValues[0].length;
    const meanAbsSHAP = Array(nFeatures).fill(0);
    const meanSHAP = Array(nFeatures).fill(0);

    for (const row of shapValues) {
        for (let f = 0; f < nFeatures; f++) {
            meanAbsSHAP[f] += Math.abs(row[f]);
            meanSHAP[f] += row[f];
        }
    }

    const n = shapValues.length;
    for (let f = 0; f < nFeatures; f++) {
        meanAbsSHAP[f] /= n;
        meanSHAP[f] /= n;
    }

    return { meanAbsSHAP, meanSHAP };
}
