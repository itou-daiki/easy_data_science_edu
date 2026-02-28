/**
 * Ridge Regression (L2-regularised linear regression).
 * Computes coefficients via: beta = (X^T X + alpha * I)^(-1) X^T y
 * @module regression/ridge
 */

/**
 * @class RidgeRegression
 * @description Linear regression with L2 penalty.
 */
export class RidgeRegression {
    /**
     * @param {Object}  params        - Configuration.
     * @param {number}  params.alpha  - Regularisation strength (default 1.0).
     */
    constructor(params = {}) {
        /** @type {number} */
        this.alpha = params.alpha ?? 1.0;
        /** @type {number[]|null} */
        this.coefficients = null;
        /** @type {number|null} */
        this.intercept = null;
        /** @type {number} */
        this.nFeatures = 0;
    }

    /**
     * Fit the ridge regression model.
     * @param {number[][]} X - 2D array [nSamples, nFeatures].
     * @param {number[]}   y - 1D array [nSamples].
     * @returns {RidgeRegression} The fitted model.
     */
    fit(X, y) {
        if (!X || !y || X.length === 0 || y.length === 0) {
            throw new Error('Training data must not be empty.');
        }
        if (X.length !== y.length) {
            throw new Error('X and y must have the same number of samples.');
        }

        this.nFeatures = X[0].length;

        // Prepend intercept column
        const XWithIntercept = X.map(row => [1, ...row]);
        const nCols = XWithIntercept[0].length;

        const Xm = math.matrix(XWithIntercept);
        const ym = math.matrix(y);

        // Build penalty matrix — do NOT penalise the intercept (index 0)
        const penaltyDiag = [0, ...Array(nCols - 1).fill(this.alpha)];
        const penalty = math.diag(penaltyDiag);

        // beta = (X^T X + alpha * I')^(-1) X^T y
        const Xt = math.transpose(Xm);
        const XtX = math.multiply(Xt, Xm);
        const regularised = math.add(XtX, penalty);
        const inv = math.inv(regularised);
        const Xty = math.multiply(Xt, ym);
        const beta = math.multiply(inv, Xty);

        const betaArray = math.flatten(beta).toArray();
        this.intercept = betaArray[0];
        this.coefficients = betaArray.slice(1);

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
     * @returns {{ alpha: number, coefficients: number[]|null, intercept: number|null }}
     */
    getParams() {
        return {
            alpha: this.alpha,
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
