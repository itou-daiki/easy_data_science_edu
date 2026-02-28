/**
 * model_selection.js - Train/test splitting, cross-validation, and grid search
 *
 * All functions are pure where feasible: input arrays are never mutated.
 * Depends on globally available `math` (math.js) and `jStat`.
 *
 * @module model_selection
 */

import {
  rSquared,
  accuracy as accuracyMetric,
  meanSquaredError,
  meanAbsoluteError,
  f1Score,
  precisionScore,
  recallScore,
} from './metrics.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Deterministic pseudo-random number generator (Mulberry32).
 * Returns a function that produces values in [0, 1).
 * @param {number} seed
 * @returns {() => number}
 */
function _createRng(seed) {
  let s = seed | 0;
  return function () {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Fisher-Yates shuffle (returns a new array, does not mutate input).
 * @param {Array} arr
 * @param {() => number} rng
 * @returns {Array}
 */
function _shuffle(arr, rng) {
  const result = [...arr];
  for (let i = result.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    const tmp = result[i];
    result[i] = result[j];
    result[j] = tmp;
  }
  return result;
}

/**
 * Select elements from an array by index.
 * @param {Array} arr
 * @param {number[]} indices
 * @returns {Array}
 */
function _selectByIndices(arr, indices) {
  return indices.map((i) => arr[i]);
}

/**
 * Validate that X and y have matching lengths.
 * @param {Array} X
 * @param {Array} y
 * @param {string} context
 */
function _validateXY(X, y, context) {
  if (!Array.isArray(X) || !Array.isArray(y)) {
    throw new Error(`${context}: X and y must be arrays`);
  }
  if (X.length === 0) {
    throw new Error(`${context}: X must not be empty`);
  }
  if (X.length !== y.length) {
    throw new Error(`${context}: X (${X.length}) and y (${y.length}) must have the same length`);
  }
}

/**
 * Look up the scoring function by name.
 * @param {string} scoring
 * @returns {(yTrue: Array, yPred: Array) => number}
 */
function _getScorer(scoring) {
  const SCORERS = {
    r2: rSquared,
    accuracy: accuracyMetric,
    mse: (yt, yp) => -meanSquaredError(yt, yp), // negate so higher is better
    neg_mse: (yt, yp) => -meanSquaredError(yt, yp),
    mae: (yt, yp) => -meanAbsoluteError(yt, yp),
    neg_mae: (yt, yp) => -meanAbsoluteError(yt, yp),
    f1: (yt, yp) => f1Score(yt, yp, 'macro'),
    precision: (yt, yp) => precisionScore(yt, yp, 'macro'),
    recall: (yt, yp) => recallScore(yt, yp, 'macro'),
  };

  const fn = SCORERS[scoring];
  if (!fn) {
    throw new Error(
      `Unknown scoring metric "${scoring}". Supported: ${Object.keys(SCORERS).join(', ')}`
    );
  }
  return fn;
}

// ===========================================================================
// trainTestSplit
// ===========================================================================

/**
 * Split arrays X and y into random train and test subsets.
 *
 * @param {Array<Array<*>>} X - Feature matrix (2D)
 * @param {Array<*>} y - Target array (1D)
 * @param {Object} [options]
 * @param {number} [options.testSize=0.2] - Fraction of data for the test set (0, 1)
 * @param {number} [options.randomState=42] - Seed for the PRNG
 * @param {boolean} [options.shuffle=true] - Whether to shuffle before splitting
 * @param {boolean} [options.stratify=false] - Stratified split (preserves class proportions in y)
 * @returns {{ XTrain: Array[], XTest: Array[], yTrain: Array, yTest: Array }}
 */
export function trainTestSplit(X, y, options = {}) {
  _validateXY(X, y, 'trainTestSplit');
  const {
    testSize = 0.2,
    randomState = 42,
    shuffle = true,
    stratify = false,
  } = options;

  if (testSize <= 0 || testSize >= 1) {
    throw new Error('trainTestSplit: testSize must be between 0 and 1 (exclusive)');
  }

  const n = X.length;
  const rng = _createRng(randomState);

  let trainIndices;
  let testIndices;

  if (stratify) {
    // Group indices by class
    const classIndices = new Map();
    for (let i = 0; i < n; i++) {
      const cls = y[i];
      if (!classIndices.has(cls)) classIndices.set(cls, []);
      classIndices.get(cls).push(i);
    }

    trainIndices = [];
    testIndices = [];

    for (const [, indices] of classIndices) {
      const shuffled = shuffle ? _shuffle(indices, rng) : [...indices];
      const nTest = Math.max(1, Math.round(shuffled.length * testSize));
      testIndices.push(...shuffled.slice(0, nTest));
      trainIndices.push(...shuffled.slice(nTest));
    }

    // Shuffle the combined indices for good measure
    if (shuffle) {
      trainIndices = _shuffle(trainIndices, rng);
      testIndices = _shuffle(testIndices, rng);
    }
  } else {
    const indices = Array.from({ length: n }, (_, i) => i);
    const ordered = shuffle ? _shuffle(indices, rng) : indices;
    const nTest = Math.max(1, Math.round(n * testSize));
    testIndices = ordered.slice(0, nTest);
    trainIndices = ordered.slice(nTest);
  }

  return {
    XTrain: _selectByIndices(X, trainIndices),
    XTest: _selectByIndices(X, testIndices),
    yTrain: _selectByIndices(y, trainIndices),
    yTest: _selectByIndices(y, testIndices),
  };
}

// ===========================================================================
// KFold
// ===========================================================================

/**
 * K-Fold cross-validation iterator.
 *
 * @example
 * const kf = new KFold({ nSplits: 5 });
 * for (const [trainIdx, testIdx] of kf.split(X)) {
 *   // ...
 * }
 */
export class KFold {
  /**
   * @param {Object} [options]
   * @param {number} [options.nSplits=5]
   * @param {boolean} [options.shuffle=true]
   * @param {number} [options.randomState=42]
   */
  constructor({ nSplits = 5, shuffle = true, randomState = 42 } = {}) {
    if (nSplits < 2) throw new Error('KFold: nSplits must be >= 2');
    /** @type {number} */
    this.nSplits = nSplits;
    /** @type {boolean} */
    this.shuffle = shuffle;
    /** @type {number} */
    this.randomState = randomState;
  }

  /**
   * Generate train/test index pairs.
   * @param {Array} X - Used only for its length
   * @yields {[number[], number[]]} [trainIndices, testIndices]
   */
  *split(X) {
    if (!Array.isArray(X) || X.length === 0) {
      throw new Error('KFold.split: X must be a non-empty array');
    }
    const n = X.length;
    if (this.nSplits > n) {
      throw new Error(`KFold.split: nSplits (${this.nSplits}) cannot exceed number of samples (${n})`);
    }

    const rng = _createRng(this.randomState);
    const indices = Array.from({ length: n }, (_, i) => i);
    const ordered = this.shuffle ? _shuffle(indices, rng) : [...indices];

    const foldSize = Math.floor(n / this.nSplits);
    const remainder = n % this.nSplits;

    let start = 0;
    for (let fold = 0; fold < this.nSplits; fold++) {
      // Distribute remainder across the first `remainder` folds
      const currentFoldSize = foldSize + (fold < remainder ? 1 : 0);
      const end = start + currentFoldSize;

      const testIdx = ordered.slice(start, end);
      const trainIdx = [...ordered.slice(0, start), ...ordered.slice(end)];

      yield [trainIdx, testIdx];
      start = end;
    }
  }
}

// ===========================================================================
// StratifiedKFold
// ===========================================================================

/**
 * Stratified K-Fold: each fold preserves the class distribution of y.
 *
 * @example
 * const skf = new StratifiedKFold({ nSplits: 5 });
 * for (const [trainIdx, testIdx] of skf.split(X, y)) {
 *   // ...
 * }
 */
export class StratifiedKFold {
  /**
   * @param {Object} [options]
   * @param {number} [options.nSplits=5]
   * @param {boolean} [options.shuffle=true]
   * @param {number} [options.randomState=42]
   */
  constructor({ nSplits = 5, shuffle = true, randomState = 42 } = {}) {
    if (nSplits < 2) throw new Error('StratifiedKFold: nSplits must be >= 2');
    /** @type {number} */
    this.nSplits = nSplits;
    /** @type {boolean} */
    this.shuffle = shuffle;
    /** @type {number} */
    this.randomState = randomState;
  }

  /**
   * Generate stratified train/test index pairs.
   * @param {Array} X - Used only for its length
   * @param {Array} y - Class labels for stratification
   * @yields {[number[], number[]]} [trainIndices, testIndices]
   */
  *split(X, y) {
    if (!Array.isArray(X) || !Array.isArray(y)) {
      throw new Error('StratifiedKFold.split: X and y must be arrays');
    }
    if (X.length !== y.length) {
      throw new Error('StratifiedKFold.split: X and y must have the same length');
    }
    if (X.length === 0) {
      throw new Error('StratifiedKFold.split: arrays must not be empty');
    }

    const rng = _createRng(this.randomState);

    // Group indices by class
    const classIndices = new Map();
    for (let i = 0; i < y.length; i++) {
      const cls = y[i];
      if (!classIndices.has(cls)) classIndices.set(cls, []);
      classIndices.get(cls).push(i);
    }

    // Validate that each class has enough samples
    for (const [cls, indices] of classIndices) {
      if (indices.length < this.nSplits) {
        throw new Error(
          `StratifiedKFold: class "${cls}" has only ${indices.length} samples, ` +
          `but nSplits=${this.nSplits} requires at least ${this.nSplits}`
        );
      }
    }

    // Shuffle within each class if requested
    const shuffledClassIndices = new Map();
    for (const [cls, indices] of classIndices) {
      shuffledClassIndices.set(cls, this.shuffle ? _shuffle(indices, rng) : [...indices]);
    }

    // For each fold, pick the corresponding slice from each class
    for (let fold = 0; fold < this.nSplits; fold++) {
      const testIdx = [];
      const trainIdx = [];

      for (const [, indices] of shuffledClassIndices) {
        const foldSize = Math.floor(indices.length / this.nSplits);
        const remainder = indices.length % this.nSplits;

        // Compute start/end for this fold within this class
        let start = 0;
        for (let f = 0; f < fold; f++) {
          start += foldSize + (f < remainder ? 1 : 0);
        }
        const currentFoldSize = foldSize + (fold < remainder ? 1 : 0);
        const end = start + currentFoldSize;

        testIdx.push(...indices.slice(start, end));
        trainIdx.push(...indices.slice(0, start), ...indices.slice(end));
      }

      yield [trainIdx, testIdx];
    }
  }
}

// ===========================================================================
// crossValidate
// ===========================================================================

/**
 * Perform k-fold cross-validation on a model.
 *
 * The model must implement:
 *   - `fit(X, y)` - Train the model
 *   - `predict(X)` - Return predictions
 *
 * A fresh model instance is created for each fold by cloning parameters.
 * If the model has a `clone()` method it is preferred; otherwise
 * the model's constructor is called with no arguments.
 *
 * @param {{ fit(X: Array[], y: Array): void, predict(X: Array[]): Array }} model
 * @param {Array<Array<*>>} X - Feature matrix
 * @param {Array<*>} y - Target array
 * @param {Object} [options]
 * @param {number} [options.cv=5] - Number of folds
 * @param {'r2'|'accuracy'|'mse'|'neg_mse'|'mae'|'neg_mae'|'f1'|'precision'|'recall'} [options.scoring='r2']
 * @returns {number[]} Array of scores, one per fold
 */
export function crossValidate(model, X, y, options = {}) {
  _validateXY(X, y, 'crossValidate');
  const { cv = 5, scoring = 'r2' } = options;

  const scorer = _getScorer(scoring);
  const kf = new KFold({ nSplits: cv, shuffle: true, randomState: 42 });

  const scores = [];
  for (const [trainIdx, testIdx] of kf.split(X)) {
    const XTrain = _selectByIndices(X, trainIdx);
    const yTrain = _selectByIndices(y, trainIdx);
    const XTest = _selectByIndices(X, testIdx);
    const yTest = _selectByIndices(y, testIdx);

    // Clone the model for each fold to avoid leaking state
    const foldModel = _cloneModel(model);
    foldModel.fit(XTrain, yTrain);
    const yPred = foldModel.predict(XTest);

    scores.push(scorer(yTest, yPred));
  }

  return scores;
}

/**
 * Attempt to clone a model. Tries clone() method first, then constructor.
 * @param {Object} model
 * @returns {Object}
 */
function _cloneModel(model) {
  if (typeof model.clone === 'function') {
    return model.clone();
  }
  // Fall back to creating a new instance via the constructor
  try {
    const Constructor = model.constructor;
    if (Constructor && Constructor !== Object) {
      return new Constructor(model.params || {});
    }
  } catch {
    // ignore
  }
  // Last resort: use the same instance (caller accepts the risk)
  return model;
}

// ===========================================================================
// gridSearch
// ===========================================================================

/**
 * Exhaustive grid search over parameter combinations.
 *
 * @param {Function} ModelClass - Constructor for the model (called as `new ModelClass(params)`)
 * @param {Object<string, Array<*>>} paramGrid - Parameter name -> array of candidate values
 * @param {Array<Array<*>>} X - Feature matrix
 * @param {Array<*>} y - Target array
 * @param {Object} [options]
 * @param {number} [options.cv=3] - Number of CV folds
 * @param {'r2'|'accuracy'|'mse'|'neg_mse'|'mae'|'neg_mae'|'f1'|'precision'|'recall'} [options.scoring='r2']
 * @returns {{
 *   bestParams: Object,
 *   bestScore: number,
 *   results: Array<{ params: Object, meanScore: number, scores: number[] }>
 * }}
 */
export function gridSearch(ModelClass, paramGrid, X, y, options = {}) {
  _validateXY(X, y, 'gridSearch');
  const { cv = 3, scoring = 'r2' } = options;

  if (typeof ModelClass !== 'function') {
    throw new Error('gridSearch: ModelClass must be a constructor function');
  }

  const paramNames = Object.keys(paramGrid);
  const paramValues = paramNames.map((k) => paramGrid[k]);
  const combinations = _cartesianProduct(paramValues);

  let bestScore = -Infinity;
  let bestParams = null;
  const results = [];

  for (const combo of combinations) {
    const params = {};
    paramNames.forEach((name, i) => {
      params[name] = combo[i];
    });

    const model = new ModelClass(params);
    const scores = crossValidate(model, X, y, { cv, scoring });
    const meanScore = scores.reduce((s, v) => s + v, 0) / scores.length;

    results.push({ params: { ...params }, meanScore, scores: [...scores] });

    if (meanScore > bestScore) {
      bestScore = meanScore;
      bestParams = { ...params };
    }
  }

  // Sort results by descending mean score
  results.sort((a, b) => b.meanScore - a.meanScore);

  return { bestParams, bestScore, results };
}

/**
 * Generate the Cartesian product of multiple arrays.
 * @param {Array<Array<*>>} arrays
 * @returns {Array<Array<*>>}
 *
 * @example
 * _cartesianProduct([[1,2],['a','b']]) => [[1,'a'],[1,'b'],[2,'a'],[2,'b']]
 */
function _cartesianProduct(arrays) {
  if (arrays.length === 0) return [[]];
  return arrays.reduce(
    (acc, curr) => {
      const expanded = [];
      for (const existing of acc) {
        for (const val of curr) {
          expanded.push([...existing, val]);
        }
      }
      return expanded;
    },
    [[]]
  );
}
