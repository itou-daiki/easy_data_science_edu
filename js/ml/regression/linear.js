/**
 * Linear Regression using Ordinary Least Squares (OLS).
 * Computes coefficients via the normal equation: beta = (X^T X)^(-1) X^T y
 * @module regression/linear
 */

/**
 * @class LinearRegression
 * @description OLS linear regression with intercept term.
 */
export class LinearRegression {
    /**
     * @param {Object} params - Configuration (reserved for future options).
     */
    constructor(params = {}) {
        /** @type {number[]|null} Feature coefficients (excluding intercept) */
        this.coefficients = null;
        /** @type {number|null} Intercept (bias) term */
        this.intercept = null;
        /** @type {number} Number of features seen during fit */
        this.nFeatures = 0;
    }

    /**
     * Fit the model using the normal equation.
     * @param {number[][]} X - 2D array of shape [nSamples, nFeatures].
     * @param {number[]}   y - 1D array of shape [nSamples].
     * @returns {LinearRegression} The fitted model instance.
     */
    fit(X, y) {
        if (!X || !y || X.length === 0 || y.length === 0) {
            throw new Error('Training data must not be empty.');
        }
        if (X.length !== y.length) {
            throw new Error('X and y must have the same number of samples.');
        }

        this.nFeatures = X[0].length;

        // Prepend a column of 1s for the intercept
        const XWithIntercept = X.map(row => [1, ...row]);

        const Xm = math.matrix(XWithIntercept);
        const ym = math.matrix(y);

        // beta = (X^T X)^(-1) X^T y
        const Xt = math.transpose(Xm);
        const XtX = math.multiply(Xt, Xm);
        const XtXInv = math.inv(XtX);
        const Xty = math.multiply(Xt, ym);
        const beta = math.multiply(XtXInv, Xty);

        const betaArray = math.flatten(beta).toArray();
        this.intercept = betaArray[0];
        this.coefficients = betaArray.slice(1);

        return this;
    }

    /**
     * Predict target values for the given feature matrix.
     * @param {number[][]} X - 2D array of shape [nSamples, nFeatures].
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
     * Return the model parameters.
     * @returns {{ coefficients: number[]|null, intercept: number|null }}
     */
    getParams() {
        return {
            coefficients: this.coefficients ? [...this.coefficients] : null,
            intercept: this.intercept,
        };
    }

    /**
     * Return feature importances based on absolute coefficient values.
     * @returns {number[]|null} Normalised importances or null if not fitted.
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
