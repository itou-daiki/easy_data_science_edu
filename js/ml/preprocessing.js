/**
 * preprocessing.js - Data preprocessing utilities for easyDataScience
 *
 * Provides scalers, encoders, imputers, and a full preprocessing pipeline.
 * All classes are immutable: fit/transform return new arrays, never mutating input.
 * Depends on globally available `math` (math.js) and `jStat`.
 *
 * @module preprocessing
 */

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Compute the arithmetic mean of a numeric array, ignoring null/undefined/NaN.
 * @param {number[]} arr
 * @returns {number}
 */
function _mean(arr) {
  const valid = arr.filter((v) => v != null && !Number.isNaN(v));
  if (valid.length === 0) return 0;
  return valid.reduce((s, v) => s + v, 0) / valid.length;
}

/**
 * Compute sample standard deviation, ignoring null/undefined/NaN.
 * Uses Bessel's correction (N-1) when N > 1.
 * @param {number[]} arr
 * @param {number} [mu] - Pre-computed mean (avoids double pass).
 * @returns {number}
 */
function _std(arr, mu) {
  const valid = arr.filter((v) => v != null && !Number.isNaN(v));
  if (valid.length <= 1) return 0;
  const m = mu !== undefined ? mu : _mean(valid);
  const variance = valid.reduce((s, v) => s + (v - m) ** 2, 0) / (valid.length - 1);
  return Math.sqrt(variance);
}

/**
 * Compute the median of a numeric array, ignoring null/undefined/NaN.
 * @param {number[]} arr
 * @returns {number}
 */
function _median(arr) {
  const sorted = arr
    .filter((v) => v != null && !Number.isNaN(v))
    .slice()
    .sort((a, b) => a - b);
  if (sorted.length === 0) return 0;
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 !== 0
    ? sorted[mid]
    : (sorted[mid - 1] + sorted[mid]) / 2;
}

/**
 * Compute the mode (most frequent value) of an array, ignoring null/undefined/NaN.
 * Ties are broken by returning the first-encountered most-frequent value.
 * @param {number[]} arr
 * @returns {number}
 */
function _mode(arr) {
  const valid = arr.filter((v) => v != null && !Number.isNaN(v));
  if (valid.length === 0) return 0;
  const freq = new Map();
  for (const v of valid) {
    freq.set(v, (freq.get(v) || 0) + 1);
  }
  let bestVal = valid[0];
  let bestCount = 0;
  for (const [val, count] of freq) {
    if (count > bestCount) {
      bestCount = count;
      bestVal = val;
    }
  }
  return bestVal;
}

/**
 * Extract a column from a 2D array.
 * @param {Array<Array<*>>} data
 * @param {number} colIdx
 * @returns {Array<*>}
 */
function _column(data, colIdx) {
  return data.map((row) => row[colIdx]);
}

/**
 * Validate that data is a non-empty 2D array.
 * @param {*} data
 * @param {string} context
 */
function _validate2D(data, context) {
  if (!Array.isArray(data) || data.length === 0) {
    throw new Error(`${context}: data must be a non-empty 2D array`);
  }
  if (!Array.isArray(data[0])) {
    throw new Error(`${context}: data must be a 2D array (array of arrays)`);
  }
}

/**
 * Compute Pearson correlation coefficient between two numeric arrays.
 * @param {number[]} a
 * @param {number[]} b
 * @returns {number} Correlation in [-1, 1], or 0 if undefined.
 */
function _pearsonCorrelation(a, b) {
  const n = a.length;
  if (n === 0) return 0;
  const meanA = a.reduce((s, v) => s + v, 0) / n;
  const meanB = b.reduce((s, v) => s + v, 0) / n;
  let num = 0, denA = 0, denB = 0;
  for (let i = 0; i < n; i++) {
    const da = a[i] - meanA;
    const db = b[i] - meanB;
    num += da * db;
    denA += da * da;
    denB += db * db;
  }
  const den = Math.sqrt(denA * denB);
  return den === 0 ? 0 : num / den;
}

// ---------------------------------------------------------------------------
// StandardScaler
// ---------------------------------------------------------------------------

/**
 * Scale features to zero mean and unit variance (z-score normalisation).
 *
 * @example
 * const scaler = new StandardScaler();
 * const scaled = scaler.fitTransform([[1,2],[3,4],[5,6]]);
 */
export class StandardScaler {
  constructor() {
    /** @type {number[]|null} Per-feature means */
    this.means = null;
    /** @type {number[]|null} Per-feature standard deviations */
    this.stds = null;
    /** @type {boolean} */
    this.isFitted = false;
  }

  /**
   * Compute per-feature mean and std from training data.
   * @param {number[][]} data - 2D array [samples x features]
   * @returns {StandardScaler} this (for chaining)
   */
  fit(data) {
    _validate2D(data, 'StandardScaler.fit');
    const nCols = data[0].length;
    const means = [];
    const stds = [];
    for (let j = 0; j < nCols; j++) {
      const col = _column(data, j);
      const m = _mean(col);
      means.push(m);
      stds.push(_std(col, m));
    }
    // Return a conceptually new state by assigning (class instances are reference-typed)
    this.means = Object.freeze(means);
    this.stds = Object.freeze(stds);
    this.isFitted = true;
    return this;
  }

  /**
   * Apply z-score scaling using previously fitted parameters.
   * @param {number[][]} data
   * @returns {number[][]} Scaled copy of data
   */
  transform(data) {
    if (!this.isFitted) throw new Error('StandardScaler: must call fit() before transform()');
    _validate2D(data, 'StandardScaler.transform');
    return data.map((row) =>
      row.map((val, j) => {
        const s = this.stds[j];
        return s === 0 ? 0 : (val - this.means[j]) / s;
      })
    );
  }

  /**
   * Fit and transform in a single step.
   * @param {number[][]} data
   * @returns {number[][]}
   */
  fitTransform(data) {
    this.fit(data);
    return this.transform(data);
  }

  /**
   * Reverse the scaling transformation.
   * @param {number[][]} data - Scaled data
   * @returns {number[][]} Data in original scale
   */
  inverseTransform(data) {
    if (!this.isFitted) throw new Error('StandardScaler: must call fit() before inverseTransform()');
    _validate2D(data, 'StandardScaler.inverseTransform');
    return data.map((row) =>
      row.map((val, j) => val * this.stds[j] + this.means[j])
    );
  }
}

// ---------------------------------------------------------------------------
// MinMaxScaler
// ---------------------------------------------------------------------------

/**
 * Scale features to the [0, 1] range using min-max normalisation.
 *
 * @example
 * const scaler = new MinMaxScaler();
 * const scaled = scaler.fitTransform([[1],[5],[3]]);
 * // scaled => [[0],[1],[0.5]]
 */
export class MinMaxScaler {
  constructor() {
    /** @type {number[]|null} */
    this.mins = null;
    /** @type {number[]|null} */
    this.maxs = null;
    /** @type {boolean} */
    this.isFitted = false;
  }

  /**
   * Compute per-feature min and max from training data.
   * @param {number[][]} data
   * @returns {MinMaxScaler}
   */
  fit(data) {
    _validate2D(data, 'MinMaxScaler.fit');
    const nCols = data[0].length;
    const mins = [];
    const maxs = [];
    for (let j = 0; j < nCols; j++) {
      const col = _column(data, j).filter((v) => v != null && !Number.isNaN(v));
      mins.push(Math.min(...col));
      maxs.push(Math.max(...col));
    }
    this.mins = Object.freeze(mins);
    this.maxs = Object.freeze(maxs);
    this.isFitted = true;
    return this;
  }

  /**
   * Transform data to [0, 1] range.
   * @param {number[][]} data
   * @returns {number[][]}
   */
  transform(data) {
    if (!this.isFitted) throw new Error('MinMaxScaler: must call fit() before transform()');
    _validate2D(data, 'MinMaxScaler.transform');
    return data.map((row) =>
      row.map((val, j) => {
        const range = this.maxs[j] - this.mins[j];
        return range === 0 ? 0 : (val - this.mins[j]) / range;
      })
    );
  }

  /**
   * Fit and transform in a single step.
   * @param {number[][]} data
   * @returns {number[][]}
   */
  fitTransform(data) {
    this.fit(data);
    return this.transform(data);
  }
}

// ---------------------------------------------------------------------------
// LabelEncoder
// ---------------------------------------------------------------------------

/**
 * Encode categorical labels as integers 0..N-1.
 *
 * @example
 * const le = new LabelEncoder();
 * le.fit(['cat','dog','cat','fish']);
 * le.transform(['dog','fish']); // [1, 2]
 * le.inverseTransform([0,2]);   // ['cat','fish']
 */
export class LabelEncoder {
  constructor() {
    /** @type {Array<*>|null} Sorted unique classes */
    this._classes = null;
    /** @type {Map<*,number>|null} */
    this._classToIndex = null;
    /** @type {boolean} */
    this.isFitted = false;
  }

  /** @type {Array<*>} Unique classes discovered during fit (read-only copy). */
  get classes() {
    if (!this.isFitted) throw new Error('LabelEncoder: must call fit() first');
    return [...this._classes];
  }

  /**
   * Discover unique classes from the provided labels.
   * Classes are sorted lexicographically for determinism.
   * @param {Array<*>} labels - 1D array of labels
   * @returns {LabelEncoder}
   */
  fit(labels) {
    if (!Array.isArray(labels) || labels.length === 0) {
      throw new Error('LabelEncoder.fit: labels must be a non-empty array');
    }
    const unique = [...new Set(labels)].sort((a, b) => String(a).localeCompare(String(b)));
    this._classes = Object.freeze(unique);
    const map = new Map();
    unique.forEach((c, i) => map.set(c, i));
    this._classToIndex = map;
    this.isFitted = true;
    return this;
  }

  /**
   * Transform labels to integer codes.
   * @param {Array<*>} labels
   * @returns {number[]}
   */
  transform(labels) {
    if (!this.isFitted) throw new Error('LabelEncoder: must call fit() before transform()');
    return labels.map((l) => {
      const idx = this._classToIndex.get(l);
      if (idx === undefined) throw new Error(`LabelEncoder.transform: unseen label "${l}"`);
      return idx;
    });
  }

  /**
   * Convert integer codes back to original labels.
   * @param {number[]} encoded
   * @returns {Array<*>}
   */
  inverseTransform(encoded) {
    if (!this.isFitted) throw new Error('LabelEncoder: must call fit() before inverseTransform()');
    return encoded.map((i) => {
      if (i < 0 || i >= this._classes.length) {
        throw new Error(`LabelEncoder.inverseTransform: index ${i} out of range`);
      }
      return this._classes[i];
    });
  }
}

// ---------------------------------------------------------------------------
// OneHotEncoder
// ---------------------------------------------------------------------------

/**
 * One-hot encode categorical labels into a 2D binary matrix.
 *
 * @example
 * const ohe = new OneHotEncoder();
 * ohe.fit(['a','b','c']);
 * ohe.transform(['b','a']); // [[0,1,0],[1,0,0]]
 */
export class OneHotEncoder {
  constructor() {
    /** @type {Array<*>|null} */
    this._classes = null;
    /** @type {Map<*,number>|null} */
    this._classToIndex = null;
    /** @type {boolean} */
    this.isFitted = false;
  }

  /** @type {Array<*>} */
  get classes() {
    if (!this.isFitted) throw new Error('OneHotEncoder: must call fit() first');
    return [...this._classes];
  }

  /**
   * Discover unique categories.
   * @param {Array<*>} labels
   * @returns {OneHotEncoder}
   */
  fit(labels) {
    if (!Array.isArray(labels) || labels.length === 0) {
      throw new Error('OneHotEncoder.fit: labels must be a non-empty array');
    }
    const unique = [...new Set(labels)].sort((a, b) => String(a).localeCompare(String(b)));
    this._classes = Object.freeze(unique);
    const map = new Map();
    unique.forEach((c, i) => map.set(c, i));
    this._classToIndex = map;
    this.isFitted = true;
    return this;
  }

  /**
   * Transform labels into a 2D one-hot matrix.
   * @param {Array<*>} labels
   * @returns {number[][]} Binary matrix [labels.length x nClasses]
   */
  transform(labels) {
    if (!this.isFitted) throw new Error('OneHotEncoder: must call fit() before transform()');
    const nClasses = this._classes.length;
    return labels.map((l) => {
      const idx = this._classToIndex.get(l);
      if (idx === undefined) throw new Error(`OneHotEncoder.transform: unseen label "${l}"`);
      const row = new Array(nClasses).fill(0);
      row[idx] = 1;
      return row;
    });
  }
}

// ---------------------------------------------------------------------------
// SimpleImputer
// ---------------------------------------------------------------------------

/**
 * Fill missing values (null, undefined, NaN) column-by-column in a 2D array.
 *
 * @example
 * const imp = new SimpleImputer({ strategy: 'mean' });
 * imp.fit([[1,2],[NaN,4],[3,NaN]]);
 * imp.transform([[NaN, NaN]]); // [[2, 3]]
 */
export class SimpleImputer {
  /**
   * @param {Object} [options]
   * @param {'mean'|'median'|'mode'} [options.strategy='mean']
   */
  constructor({ strategy = 'mean' } = {}) {
    const VALID_STRATEGIES = ['mean', 'median', 'mode'];
    if (!VALID_STRATEGIES.includes(strategy)) {
      throw new Error(`SimpleImputer: strategy must be one of ${VALID_STRATEGIES.join(', ')}`);
    }
    /** @type {'mean'|'median'|'mode'} */
    this.strategy = strategy;
    /** @type {number[]|null} Per-column fill values */
    this.fillValues = null;
    /** @type {boolean} */
    this.isFitted = false;
  }

  /**
   * Compute the fill value for each column.
   * @param {number[][]} data - 2D array
   * @returns {SimpleImputer}
   */
  fit(data) {
    _validate2D(data, 'SimpleImputer.fit');
    const nCols = data[0].length;
    const fillValues = [];
    const strategyFn =
      this.strategy === 'mean' ? _mean : this.strategy === 'median' ? _median : _mode;

    for (let j = 0; j < nCols; j++) {
      fillValues.push(strategyFn(_column(data, j)));
    }
    this.fillValues = Object.freeze(fillValues);
    this.isFitted = true;
    return this;
  }

  /**
   * Replace missing values with the fitted fill values.
   * @param {number[][]} data
   * @returns {number[][]} New 2D array with missing values filled
   */
  transform(data) {
    if (!this.isFitted) throw new Error('SimpleImputer: must call fit() before transform()');
    _validate2D(data, 'SimpleImputer.transform');
    return data.map((row) =>
      row.map((val, j) =>
        val == null || Number.isNaN(val) ? this.fillValues[j] : val
      )
    );
  }

  /**
   * Fit and transform in a single step.
   * @param {number[][]} data
   * @returns {number[][]}
   */
  fitTransform(data) {
    this.fit(data);
    return this.transform(data);
  }
}

// ---------------------------------------------------------------------------
// encodeCategorials
// ---------------------------------------------------------------------------

/**
 * Automatically label-encode specified categorical columns in a 2D dataset.
 *
 * @param {Array<Array<*>>} data - 2D array [samples x features]
 * @param {number[]} columns - Column indices to encode
 * @returns {{ encodedData: Array<Array<*>>, encoders: Map<number, LabelEncoder> }}
 */
export function encodeCategorials(data, columns) {
  _validate2D(data, 'encodeCategorials');
  if (!Array.isArray(columns)) {
    throw new Error('encodeCategorials: columns must be an array of column indices');
  }

  /** @type {Map<number, LabelEncoder>} */
  const encoders = new Map();

  // Fit encoders for each categorical column
  for (const colIdx of columns) {
    const colValues = _column(data, colIdx);
    const encoder = new LabelEncoder();
    encoder.fit(colValues.filter((v) => v != null));
    encoders.set(colIdx, encoder);
  }

  // Produce a new dataset with encoded columns
  const encodedData = data.map((row) => {
    const newRow = [...row];
    for (const colIdx of columns) {
      const encoder = encoders.get(colIdx);
      const val = newRow[colIdx];
      if (val != null) {
        newRow[colIdx] = encoder.transform([val])[0];
      }
    }
    return newRow;
  });

  return { encodedData, encoders };
}

// ---------------------------------------------------------------------------
// prepareFeatures
// ---------------------------------------------------------------------------

/**
 * Full preprocessing pipeline: separate target, impute, encode, and scale.
 *
 * @param {Array<Array<*>>} rawData - 2D array including headers as the first row
 * @param {string} targetCol - Name of the target column (must match a header value)
 * @param {Object} [options]
 * @param {'mean'|'median'|'mode'} [options.imputeStrategy='mean'] - Strategy for numeric imputation
 * @param {boolean} [options.scale=true] - Whether to standard-scale numeric features
 * @returns {{
 *   X: number[][],
 *   y: Array<*>,
 *   featureNames: string[],
 *   encoders: Map<number, LabelEncoder>,
 *   scaler: StandardScaler|null
 * }}
 */
export function prepareFeatures(rawData, targetCol, options = {}) {
  const {
    imputeStrategy = 'mean',
    scale = true,
    selectedFeatures = null,
    task = 'regression',
    removeOutliers = true,
    removeMulticollinearity = true,
    multicollinearityThreshold = 0.95,
    transformFeatures = true,
    skewnessThreshold = 2.0
  } = options;

  // --- 0. Handle array-of-objects format (from CSV/XLSX parsing) ------------
  let headers, body;
  if (rawData.length > 0 && typeof rawData[0] === 'object' && !Array.isArray(rawData[0])) {
    headers = Object.keys(rawData[0]);
    body = rawData.map(row => headers.map(h => row[h]));
  } else {
    _validate2D(rawData, 'prepareFeatures');
    if (rawData.length < 2) {
      throw new Error('prepareFeatures: rawData must have a header row and at least one data row');
    }
    headers = rawData[0].map(String);
    body = rawData.slice(1);
  }

  const targetIdx = headers.indexOf(targetCol);
  if (targetIdx === -1) {
    throw new Error(`prepareFeatures: target column "${targetCol}" not found in headers`);
  }

  // --- 1. Filter to selected features if specified --------------------------
  let featureIndices = headers.map((_, i) => i).filter((i) => i !== targetIdx);
  if (selectedFeatures && Array.isArray(selectedFeatures) && selectedFeatures.length > 0) {
    featureIndices = featureIndices.filter(i => selectedFeatures.includes(headers[i]));
  }
  const featureNames = featureIndices.map((i) => headers[i]);

  // --- 2. Separate target and features --------------------------------------
  const yRaw = body.map((row) => row[targetIdx]);
  let X = body.map((row) => featureIndices.map((i) => row[i]));

  // --- 2b. Encode target for classification ---------------------------------
  let y;
  let labelEncoder = null;
  if (task === 'classification') {
    const isTargetNumeric = yRaw.every(v => v != null && !isNaN(Number(v)));
    if (isTargetNumeric) {
      y = yRaw.map(Number);
    } else {
      labelEncoder = new LabelEncoder();
      labelEncoder.fit(yRaw.filter(v => v != null));
      y = labelEncoder.transform(yRaw);
    }
  } else {
    y = yRaw.map(v => v != null ? Number(v) : null);
  }

  // --- 3. Identify categorical vs numeric columns ---------------------------
  const categoricalCols = [];
  const numericCols = [];

  for (let j = 0; j < featureNames.length; j++) {
    const colValues = _column(X, j).filter((v) => v != null);
    const isNumeric = colValues.length > 0 && colValues.every((v) => typeof v === 'number' || !Number.isNaN(Number(v)));
    if (isNumeric) {
      numericCols.push(j);
    } else {
      categoricalCols.push(j);
    }
  }

  // --- 4. Convert numeric string values to numbers --------------------------
  X = X.map((row) =>
    row.map((val, j) => {
      if (numericCols.includes(j) && val != null) {
        const n = Number(val);
        return Number.isNaN(n) ? null : n;
      }
      return val;
    })
  );

  // --- Preprocessing info tracking -------------------------------------------
  const preprocessInfo = {
    outlierRows: 0,
    removedMulticollinear: [],
    transformedFeatures: []
  };

  // --- 5. Impute missing values for numeric columns -------------------------
  if (numericCols.length > 0) {
    const numericData = X.map((row) => numericCols.map((j) => row[j]));
    const imputer = new SimpleImputer({ strategy: imputeStrategy });
    const imputedNumeric = imputer.fitTransform(numericData);

    X = X.map((row, i) => {
      const newRow = [...row];
      numericCols.forEach((j, k) => {
        newRow[j] = imputedNumeric[i][k];
      });
      return newRow;
    });
  }

  // --- 5.5 Remove outliers (IQR method) ------------------------------------
  if (removeOutliers && numericCols.length > 0) {
    const keepMask = Array(X.length).fill(true);
    for (const j of numericCols) {
      const vals = X.map(row => row[j]).filter(v => v != null && !Number.isNaN(v));
      if (vals.length < 10) continue; // skip columns with too few values
      const sorted = [...vals].sort((a, b) => a - b);
      const q1 = sorted[Math.floor(sorted.length * 0.25)];
      const q3 = sorted[Math.floor(sorted.length * 0.75)];
      const iqr = q3 - q1;
      if (iqr === 0) continue;
      const lower = q1 - 1.5 * iqr;
      const upper = q3 + 1.5 * iqr;
      for (let i = 0; i < X.length; i++) {
        const v = X[i][j];
        if (v != null && (v < lower || v > upper)) {
          keepMask[i] = false;
        }
      }
    }
    const originalLen = X.length;
    X = X.filter((_, i) => keepMask[i]);
    y = y.filter((_, i) => keepMask[i]);
    preprocessInfo.outlierRows = originalLen - X.length;
  }

  // --- 5.6 Feature transformation (log for skewed features) ----------------
  if (transformFeatures && numericCols.length > 0) {
    for (const j of numericCols) {
      const vals = X.map(row => row[j]).filter(v => v != null && !Number.isNaN(v));
      if (vals.length < 5) continue;
      const n = vals.length;
      const mu = vals.reduce((s, v) => s + v, 0) / n;
      const m3 = vals.reduce((s, v) => s + Math.pow(v - mu, 3), 0) / n;
      const m2 = vals.reduce((s, v) => s + Math.pow(v - mu, 2), 0) / n;
      const stdDev = Math.sqrt(m2);
      if (stdDev === 0) continue;
      const skewness = m3 / Math.pow(stdDev, 3);
      // Apply log1p for positively skewed features with all non-negative values
      if (Math.abs(skewness) > skewnessThreshold) {
        const minVal = Math.min(...vals);
        if (minVal >= 0) {
          X = X.map(row => {
            const newRow = [...row];
            newRow[j] = Math.log1p(newRow[j]);
            return newRow;
          });
          preprocessInfo.transformedFeatures.push(featureNames[j]);
        }
      }
    }
  }

  // --- 6. Encode categorical columns ----------------------------------------
  let encoders = new Map();
  if (categoricalCols.length > 0) {
    const catData = X.map((row) => categoricalCols.map((j) => row[j]));
    for (let k = 0; k < categoricalCols.length; k++) {
      const colVals = _column(catData, k);
      const modeVal = _modeCategorical(colVals);
      X = X.map((row) => {
        const newRow = [...row];
        if (newRow[categoricalCols[k]] == null) {
          newRow[categoricalCols[k]] = modeVal;
        }
        return newRow;
      });
    }

    const result = encodeCategorials(X, categoricalCols);
    X = result.encodedData;
    encoders = result.encoders;
  }

  // Ensure all values are numbers at this point
  X = X.map((row) => row.map((v) => (typeof v === 'number' ? v : Number(v))));

  // --- 6.5 Remove multicollinear features ----------------------------------
  if (removeMulticollinearity && X.length > 0 && X[0].length > 1) {
    const nFeats = X[0].length;
    const colsToRemove = new Set();
    for (let a = 0; a < nFeats; a++) {
      if (colsToRemove.has(a)) continue;
      for (let b = a + 1; b < nFeats; b++) {
        if (colsToRemove.has(b)) continue;
        const colA = X.map(row => row[a]);
        const colB = X.map(row => row[b]);
        const corr = _pearsonCorrelation(colA, colB);
        if (Math.abs(corr) > multicollinearityThreshold) {
          colsToRemove.add(b);
          preprocessInfo.removedMulticollinear.push(featureNames[b]);
        }
      }
    }
    if (colsToRemove.size > 0) {
      const keepCols = Array.from({ length: nFeats }, (_, i) => i).filter(i => !colsToRemove.has(i));
      X = X.map(row => keepCols.map(i => row[i]));
      const newFeatureNames = keepCols.map(i => featureNames[i]);
      // Update encoders keys
      const newEncoders = new Map();
      for (const [oldIdx, enc] of encoders) {
        const newIdx = keepCols.indexOf(oldIdx);
        if (newIdx !== -1) newEncoders.set(newIdx, enc);
      }
      featureNames.length = 0;
      featureNames.push(...newFeatureNames);
      encoders = newEncoders;
    }
  }

  // --- 7. Scale numeric features --------------------------------------------
  let scaler = null;
  if (scale) {
    scaler = new StandardScaler();
    X = scaler.fitTransform(X);
  }

  return { X, y, featureNames, encoders, scaler, labelEncoder, preprocessInfo };
}

/**
 * Mode for categorical (string) values.
 * @param {Array<*>} arr
 * @returns {*}
 */
function _modeCategorical(arr) {
  const valid = arr.filter((v) => v != null);
  if (valid.length === 0) return null;
  const freq = new Map();
  for (const v of valid) {
    freq.set(v, (freq.get(v) || 0) + 1);
  }
  let bestVal = valid[0];
  let bestCount = 0;
  for (const [val, count] of freq) {
    if (count > bestCount) {
      bestCount = count;
      bestVal = val;
    }
  }
  return bestVal;
}
