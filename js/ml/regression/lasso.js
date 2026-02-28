/**
 * Lasso Regression (L1-regularised linear regression).
 * Uses coordinate descent to minimise: (1/2n)||y - Xw||^2 + alpha * ||w||_1
 * @module regression/lasso
 */

/**
 * @class LassoRegression
 * @description Linear regression with L1 penalty via coordinate descent.
 */
export class LassoRegression {
    /**
     * @param {Object}  params          - Configuration.
     * @param {number}  params.alpha    - Regularisation strength (default 1.0).
     * @param {number}  params.maxIter  - Max iterations (default 1000).
     * @param {number}  params.tol      - Convergence tolerance (default 1e-4).
     */
    constructor(params = {}) {
        /** @type {number} */
        this.alpha = params.alpha ?? 1.0;
        /** @type {number} */
        this.maxIter = params.maxIter ?? 1000;
        /** @type {number} */
        this.tol = params.tol ?? 1e-4;
        /** @type {number[]|null} */
        this.coefficients = null;
        /** @type {number|null} */
        this.intercept = null;
        /** @type {number} */
        this.nFeatures = 0;
    }

    /**
     * Soft-thresholding operator.
     * @param {number} rho - Coordinate-wise gradient value.
     * @param {number} lam - Penalty parameter.
     * @returns {number}
     */
    static _softThreshold(rho, lam) {
        if (rho > lam) return rho - lam;
        if (rho < -lam) return rho + lam;
        return 0;
    }

    /**
     * Fit using coordinate descent.
     * @param {number[][]} X - 2D array [nSamples, nFeatures].
     * @param {number[]}   y - 1D array [nSamples].
     * @returns {LassoRegression} The fitted model.
     */
    fit(X, y) {
        if (!X || !y || X.length === 0 || y.length === 0) {
            throw new Error('Training data must not be empty.');
        }
        if (X.length !== y.length) {
            throw new Error('X and y must have the same number of samples.');
        }

        const n = X.length;
        const p = X[0].length;
        this.nFeatures = p;

        // Centre y to handle intercept separately
        const yMean = y.reduce((a, b) => a + b, 0) / n;

        // Column means for centring X
        const xMeans = Array(p).fill(0);
        for (let j = 0; j < p; j++) {
            for (let i = 0; i < n; i++) {
                xMeans[j] += X[i][j];
            }
            xMeans[j] /= n;
        }

        // Build centred copies (immutable approach — original X/y untouched)
        const Xc = X.map(row => row.map((v, j) => v - xMeans[j]));
        const yc = y.map(v => v - yMean);

        // Pre-compute column squared norms
        const colNorms = Array(p).fill(0);
        for (let j = 0; j < p; j++) {
            for (let i = 0; i < n; i++) {
                colNorms[j] += Xc[i][j] * Xc[i][j];
            }
        }

        // Initialise coefficients to zero
        let w = Array(p).fill(0);
        const residual = [...yc];

        for (let iter = 0; iter < this.maxIter; iter++) {
            let maxChange = 0;

            for (let j = 0; j < p; j++) {
                const oldW = w[j];

                // Add back contribution of feature j to residual
                if (oldW !== 0) {
                    for (let i = 0; i < n; i++) {
                        residual[i] += Xc[i][j] * oldW;
                    }
                }

                // Compute rho = X_j^T * residual
                let rho = 0;
                for (let i = 0; i < n; i++) {
                    rho += Xc[i][j] * residual[i];
                }

                // Update with soft thresholding
                const newW = colNorms[j] === 0
                    ? 0
                    : LassoRegression._softThreshold(rho, this.alpha * n) / colNorms[j];
                w[j] = newW;

                // Update residual with new contribution
                if (newW !== 0) {
                    for (let i = 0; i < n; i++) {
                        residual[i] -= Xc[i][j] * newW;
                    }
                }

                maxChange = Math.max(maxChange, Math.abs(newW - oldW));
            }

            if (maxChange < this.tol) {
                break;
            }
        }

        this.coefficients = [...w];
        // Recover intercept: intercept = yMean - sum(xMeans[j] * w[j])
        this.intercept = yMean - xMeans.reduce((acc, m, j) => acc + m * w[j], 0);

        return this;
    }

    /**
     * Predict target values.
     * @param {number[][]} X - 2D array [nSamples, nFeatures].
     * @returns {number[]} Predicted values.
     */
    predict(X) {
        if (!this.coefficients) {
            throw new Error('Model has not been fitted yet. Call fit() first.');
        }
        if (!X || X.length === 0) {
            return [];
        }

        const coeffs = this.coefficients;
        const intercept = this.intercept;

        return X.map(row => {
            let sum = intercept;
            for (let j = 0; j < coeffs.length; j++) {
                sum += coeffs[j] * row[j];
            }
            return sum;
        });
    }

    /**
     * Return model parameters.
     * @returns {{ alpha: number, maxIter: number, tol: number, coefficients: number[]|null, intercept: number|null }}
     */
    getParams() {
        return {
            alpha: this.alpha,
            maxIter: this.maxIter,
            tol: this.tol,
            coefficients: this.coefficients ? [...this.coefficients] : null,
            intercept: this.intercept,
        };
    }

    /**
     * Return feature importances (normalised absolute coefficients).
     * @returns {number[]|null}
     */
    getFeatureImportance() {
        if (!this.coefficients) {
            return null;
        }
        const absCoeffs = this.coefficients.map(Math.abs);
        const total = absCoeffs.reduce((a, b) => a + b, 0);
        if (total === 0) {
            return this.coefficients.map(() => 0);
        }
        return absCoeffs.map(v => v / total);
    }
}
