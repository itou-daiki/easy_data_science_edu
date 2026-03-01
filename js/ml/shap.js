/**
 * shap.js - SHAP (SHapley Additive exPlanations) value computation
 *
 * Provides two methods:
 * 1. linearSHAP - Exact SHAP for linear models (O(n*p))
 * 2. kernelSHAP - Model-agnostic approximate SHAP (O(2^p * bg * n))
 *
 * @module shap
 */

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
 * Compute approximate SHAP values using Kernel SHAP.
 *
 * Uses the Shapley kernel weighting scheme with weighted least squares
 * to estimate feature contributions for any model.
 *
 * @param {function(number[][]): number[]} predictFn - Prediction function (batch input → batch output)
 * @param {number[][]} X - Instances to explain (n x p)
 * @param {number[][]} background - Background dataset for marginalization
 * @param {Object} [options]
 * @param {number} [options.maxBackground=50] - Max background samples to use
 * @returns {{ shapValues: number[][], baseValue: number }}
 */
export function kernelSHAP(predictFn, X, background, options = {}) {
    const { maxBackground = 50 } = options;

    const bg = background.length > maxBackground
        ? background.slice(0, maxBackground)
        : background;

    const bgPreds = predictFn(bg);
    const baseValue = bgPreds.reduce((a, b) => a + b, 0) / bgPreds.length;

    const nFeatures = X[0].length;

    const shapValues = X.map(x =>
        _kernelSHAPInstance(predictFn, x, bg, nFeatures, baseValue)
    );

    return { shapValues, baseValue };
}

/**
 * Compute SHAP values for a single instance using Kernel SHAP.
 * @private
 */
function _kernelSHAPInstance(predictFn, x, background, nFeatures, baseValue) {
    const totalCoalitions = 1 << nFeatures; // 2^nFeatures
    const coalitions = [];
    const weights = [];
    const effects = [];

    for (let mask = 1; mask < totalCoalitions - 1; mask++) {
        const coalition = [];
        let nIncluded = 0;
        for (let f = 0; f < nFeatures; f++) {
            const included = (mask >> f) & 1;
            coalition.push(included);
            nIncluded += included;
        }

        const weight = _kernelWeight(nFeatures, nIncluded);

        // Build masked samples: included features use x, excluded use background
        const maskedX = background.map(bg =>
            x.map((val, f) => coalition[f] ? val : bg[f])
        );

        const preds = predictFn(maskedX);
        const meanPred = preds.reduce((a, b) => a + b, 0) / preds.length;

        coalitions.push(coalition);
        weights.push(weight);
        effects.push(meanPred - baseValue);
    }

    return _weightedLeastSquares(coalitions, effects, weights, nFeatures);
}

/**
 * Kernel SHAP weight: π(z) = (M-1) / (C(M,|z|) * |z| * (M-|z|))
 * where M = number of features, |z| = number of included features
 * @private
 */
function _kernelWeight(M, nIncluded) {
    return (M - 1) / (_binomial(M, nIncluded) * nIncluded * (M - nIncluded));
}

/**
 * Binomial coefficient C(n, k)
 * @private
 */
function _binomial(n, k) {
    if (k < 0 || k > n) return 0;
    if (k === 0 || k === n) return 1;
    let result = 1;
    for (let i = 0; i < Math.min(k, n - k); i++) {
        result = result * (n - i) / (i + 1);
    }
    return Math.round(result);
}

/**
 * Solve weighted least squares: (Z^T W Z) beta = Z^T W y
 * using Gaussian elimination with partial pivoting.
 * @private
 * @param {number[][]} Z - Coalition matrix (nCoalitions x nFeatures)
 * @param {number[]} y - Effects vector (nCoalitions)
 * @param {number[]} w - Weights vector (nCoalitions)
 * @param {number} nFeatures
 * @returns {number[]} SHAP values (nFeatures)
 */
function _weightedLeastSquares(Z, y, w, nFeatures) {
    // Compute Z^T W Z and Z^T W y
    const A = Array.from({ length: nFeatures }, () => Array(nFeatures).fill(0));
    const b = Array(nFeatures).fill(0);

    for (let c = 0; c < Z.length; c++) {
        for (let i = 0; i < nFeatures; i++) {
            for (let j = 0; j < nFeatures; j++) {
                A[i][j] += w[c] * Z[c][i] * Z[c][j];
            }
            b[i] += w[c] * Z[c][i] * y[c];
        }
    }

    // Add small regularization for numerical stability
    for (let i = 0; i < nFeatures; i++) {
        A[i][i] += 1e-10;
    }

    return _solveLinearSystem(A, b);
}

/**
 * Solve Ax = b using Gaussian elimination with partial pivoting.
 * @private
 * @param {number[][]} A - Square matrix (n x n)
 * @param {number[]} b - Right-hand side (n)
 * @returns {number[]} Solution vector (n)
 */
function _solveLinearSystem(A, b) {
    const n = A.length;
    // Augmented matrix [A|b]
    const aug = A.map((row, i) => [...row, b[i]]);

    // Forward elimination with partial pivoting
    for (let col = 0; col < n; col++) {
        // Find pivot
        let maxVal = Math.abs(aug[col][col]);
        let maxRow = col;
        for (let row = col + 1; row < n; row++) {
            if (Math.abs(aug[row][col]) > maxVal) {
                maxVal = Math.abs(aug[row][col]);
                maxRow = row;
            }
        }
        // Swap rows
        if (maxRow !== col) {
            const tmp = aug[col];
            aug[col] = aug[maxRow];
            aug[maxRow] = tmp;
        }

        const pivot = aug[col][col];
        if (Math.abs(pivot) < 1e-12) continue;

        // Eliminate below
        for (let row = col + 1; row < n; row++) {
            const factor = aug[row][col] / pivot;
            for (let j = col; j <= n; j++) {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }

    // Back substitution
    const x = Array(n).fill(0);
    for (let i = n - 1; i >= 0; i--) {
        let sum = aug[i][n];
        for (let j = i + 1; j < n; j++) {
            sum -= aug[i][j] * x[j];
        }
        x[i] = Math.abs(aug[i][i]) > 1e-12 ? sum / aug[i][i] : 0;
    }

    return x;
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
