/**
 * metrics.js - Evaluation metrics for regression and classification
 *
 * All functions are pure: they never mutate their inputs.
 * Depends on globally available `math` (math.js) and `jStat`.
 *
 * @module metrics
 */

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Validate that two arrays have the same length and are non-empty.
 * @param {Array} yTrue
 * @param {Array} yPred
 * @param {string} context
 */
function _validatePair(yTrue, yPred, context) {
  if (!Array.isArray(yTrue) || !Array.isArray(yPred)) {
    throw new Error(`${context}: yTrue and yPred must be arrays`);
  }
  if (yTrue.length === 0) {
    throw new Error(`${context}: arrays must not be empty`);
  }
  if (yTrue.length !== yPred.length) {
    throw new Error(`${context}: yTrue (${yTrue.length}) and yPred (${yPred.length}) must have the same length`);
  }
}

/**
 * Discover sorted unique classes from one or two label arrays.
 * @param  {...Array<*>} arrays
 * @returns {Array<*>}
 */
function _uniqueClasses(...arrays) {
  const all = new Set();
  for (const arr of arrays) {
    for (const v of arr) all.add(v);
  }
  return [...all].sort((a, b) => String(a).localeCompare(String(b)));
}

// ===========================================================================
// REGRESSION METRICS
// ===========================================================================

/**
 * Mean Absolute Error.
 * MAE = (1/n) * sum(|yTrue_i - yPred_i|)
 *
 * @param {number[]} yTrue - Ground truth values
 * @param {number[]} yPred - Predicted values
 * @returns {number}
 */
export function meanAbsoluteError(yTrue, yPred) {
  _validatePair(yTrue, yPred, 'meanAbsoluteError');
  const n = yTrue.length;
  let sum = 0;
  for (let i = 0; i < n; i++) {
    sum += Math.abs(yTrue[i] - yPred[i]);
  }
  return sum / n;
}

/**
 * Mean Squared Error.
 * MSE = (1/n) * sum((yTrue_i - yPred_i)^2)
 *
 * @param {number[]} yTrue
 * @param {number[]} yPred
 * @returns {number}
 */
export function meanSquaredError(yTrue, yPred) {
  _validatePair(yTrue, yPred, 'meanSquaredError');
  const n = yTrue.length;
  let sum = 0;
  for (let i = 0; i < n; i++) {
    sum += (yTrue[i] - yPred[i]) ** 2;
  }
  return sum / n;
}

/**
 * Root Mean Squared Error.
 * RMSE = sqrt(MSE)
 *
 * @param {number[]} yTrue
 * @param {number[]} yPred
 * @returns {number}
 */
export function rootMeanSquaredError(yTrue, yPred) {
  return Math.sqrt(meanSquaredError(yTrue, yPred));
}

/**
 * Coefficient of Determination (R-squared).
 * R^2 = 1 - SS_res / SS_tot
 *
 * Returns negative infinity when SS_tot is zero (constant target).
 *
 * @param {number[]} yTrue
 * @param {number[]} yPred
 * @returns {number}
 */
export function rSquared(yTrue, yPred) {
  _validatePair(yTrue, yPred, 'rSquared');
  const n = yTrue.length;
  const mean = yTrue.reduce((s, v) => s + v, 0) / n;

  let ssTot = 0;
  let ssRes = 0;
  for (let i = 0; i < n; i++) {
    ssTot += (yTrue[i] - mean) ** 2;
    ssRes += (yTrue[i] - yPred[i]) ** 2;
  }

  if (ssTot === 0) return -Infinity;
  return 1 - ssRes / ssTot;
}

/**
 * Adjusted R-squared.
 * Adjusted R^2 = 1 - ((1 - R^2) * (n - 1)) / (n - p - 1)
 *
 * @param {number[]} yTrue
 * @param {number[]} yPred
 * @param {number} numFeatures - Number of predictors (p)
 * @returns {number}
 */
export function adjustedRSquared(yTrue, yPred, numFeatures) {
  if (typeof numFeatures !== 'number' || numFeatures < 1) {
    throw new Error('adjustedRSquared: numFeatures must be a positive integer');
  }
  const n = yTrue.length;
  if (n - numFeatures - 1 <= 0) {
    throw new Error('adjustedRSquared: not enough samples for the given number of features');
  }
  const r2 = rSquared(yTrue, yPred);
  return 1 - ((1 - r2) * (n - 1)) / (n - numFeatures - 1);
}

/**
 * Mean Absolute Percentage Error.
 * MAPE = (100/n) * sum(|yTrue_i - yPred_i| / |yTrue_i|)
 *
 * Samples where yTrue_i == 0 are excluded to avoid division by zero.
 * Returns NaN if all samples are excluded.
 *
 * @param {number[]} yTrue
 * @param {number[]} yPred
 * @returns {number} Percentage (0-100+)
 */
export function meanAbsolutePercentageError(yTrue, yPred) {
  _validatePair(yTrue, yPred, 'meanAbsolutePercentageError');
  let sum = 0;
  let count = 0;
  for (let i = 0; i < yTrue.length; i++) {
    if (yTrue[i] !== 0) {
      sum += Math.abs((yTrue[i] - yPred[i]) / yTrue[i]);
      count++;
    }
  }
  if (count === 0) return NaN;
  return (sum / count) * 100;
}

/**
 * Root Mean Squared Logarithmic Error.
 * RMSLE = sqrt((1/n) * sum((log(1 + yPred_i) - log(1 + yTrue_i))^2))
 *
 * Values below zero are clipped to zero before the log transform.
 *
 * @param {number[]} yTrue
 * @param {number[]} yPred
 * @returns {number}
 */
export function rootMeanSquaredLogError(yTrue, yPred) {
  _validatePair(yTrue, yPred, 'rootMeanSquaredLogError');
  const n = yTrue.length;
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const trueClipped = Math.max(0, yTrue[i]);
    const predClipped = Math.max(0, yPred[i]);
    sum += (Math.log1p(predClipped) - Math.log1p(trueClipped)) ** 2;
  }
  return Math.sqrt(sum / n);
}

// ===========================================================================
// CLASSIFICATION METRICS
// ===========================================================================

/**
 * Overall accuracy.
 * accuracy = (number of correct predictions) / n
 *
 * @param {Array<*>} yTrue
 * @param {Array<*>} yPred
 * @returns {number} Value in [0, 1]
 */
export function accuracy(yTrue, yPred) {
  _validatePair(yTrue, yPred, 'accuracy');
  let correct = 0;
  for (let i = 0; i < yTrue.length; i++) {
    if (yTrue[i] === yPred[i]) correct++;
  }
  return correct / yTrue.length;
}

/**
 * Build the confusion matrix as a 2D array.
 * Rows = true classes, Columns = predicted classes.
 * Classes are sorted lexicographically.
 *
 * @param {Array<*>} yTrue
 * @param {Array<*>} yPred
 * @returns {{ matrix: number[][], labels: Array<*> }}
 */
export function confusionMatrix(yTrue, yPred) {
  _validatePair(yTrue, yPred, 'confusionMatrix');
  const labels = _uniqueClasses(yTrue, yPred);
  const labelIdx = new Map(labels.map((l, i) => [l, i]));
  const n = labels.length;

  const matrix = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < yTrue.length; i++) {
    const trueIdx = labelIdx.get(yTrue[i]);
    const predIdx = labelIdx.get(yPred[i]);
    matrix[trueIdx][predIdx]++;
  }

  return { matrix, labels };
}

/**
 * Per-class precision, recall, f1 computed from a confusion matrix row/column.
 * @param {number[][]} cm - Confusion matrix
 * @param {number} classIdx
 * @returns {{ precision: number, recall: number, f1: number }}
 */
function _perClassMetrics(cm, classIdx) {
  const n = cm.length;
  // True positives
  const tp = cm[classIdx][classIdx];
  // False positives: column sum minus TP
  let fp = 0;
  for (let r = 0; r < n; r++) fp += cm[r][classIdx];
  fp -= tp;
  // False negatives: row sum minus TP
  let fn = 0;
  for (let c = 0; c < n; c++) fn += cm[classIdx][c];
  fn -= tp;

  const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
  const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
  const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;

  return { precision, recall, f1 };
}

/**
 * Precision score.
 *
 * @param {Array<*>} yTrue
 * @param {Array<*>} yPred
 * @param {'macro'|'micro'|'weighted'} [average='macro']
 * @returns {number}
 */
export function precisionScore(yTrue, yPred, average = 'macro') {
  _validatePair(yTrue, yPred, 'precisionScore');
  const { matrix, labels } = confusionMatrix(yTrue, yPred);

  if (average === 'micro') {
    return _microMetric(matrix).precision;
  }

  const perClass = labels.map((_, i) => _perClassMetrics(matrix, i));

  if (average === 'macro') {
    return perClass.reduce((s, m) => s + m.precision, 0) / labels.length;
  }

  // weighted
  const support = labels.map((_, i) => matrix[i].reduce((s, v) => s + v, 0));
  const total = support.reduce((s, v) => s + v, 0);
  return total === 0
    ? 0
    : perClass.reduce((s, m, i) => s + m.precision * support[i], 0) / total;
}

/**
 * Recall score.
 *
 * @param {Array<*>} yTrue
 * @param {Array<*>} yPred
 * @param {'macro'|'micro'|'weighted'} [average='macro']
 * @returns {number}
 */
export function recallScore(yTrue, yPred, average = 'macro') {
  _validatePair(yTrue, yPred, 'recallScore');
  const { matrix, labels } = confusionMatrix(yTrue, yPred);

  if (average === 'micro') {
    return _microMetric(matrix).recall;
  }

  const perClass = labels.map((_, i) => _perClassMetrics(matrix, i));

  if (average === 'macro') {
    return perClass.reduce((s, m) => s + m.recall, 0) / labels.length;
  }

  const support = labels.map((_, i) => matrix[i].reduce((s, v) => s + v, 0));
  const total = support.reduce((s, v) => s + v, 0);
  return total === 0
    ? 0
    : perClass.reduce((s, m, i) => s + m.recall * support[i], 0) / total;
}

/**
 * F1 score (harmonic mean of precision and recall).
 *
 * @param {Array<*>} yTrue
 * @param {Array<*>} yPred
 * @param {'macro'|'micro'|'weighted'} [average='macro']
 * @returns {number}
 */
export function f1Score(yTrue, yPred, average = 'macro') {
  _validatePair(yTrue, yPred, 'f1Score');
  const { matrix, labels } = confusionMatrix(yTrue, yPred);

  if (average === 'micro') {
    return _microMetric(matrix).f1;
  }

  const perClass = labels.map((_, i) => _perClassMetrics(matrix, i));

  if (average === 'macro') {
    return perClass.reduce((s, m) => s + m.f1, 0) / labels.length;
  }

  const support = labels.map((_, i) => matrix[i].reduce((s, v) => s + v, 0));
  const total = support.reduce((s, v) => s + v, 0);
  return total === 0
    ? 0
    : perClass.reduce((s, m, i) => s + m.f1 * support[i], 0) / total;
}

/**
 * Micro-averaged precision, recall, and F1 from a confusion matrix.
 * For micro averaging: precision = recall = F1 = accuracy.
 * @param {number[][]} cm
 * @returns {{ precision: number, recall: number, f1: number }}
 */
function _microMetric(cm) {
  const n = cm.length;
  let tpSum = 0;
  let fpSum = 0;
  let fnSum = 0;

  for (let c = 0; c < n; c++) {
    const tp = cm[c][c];
    tpSum += tp;
    for (let r = 0; r < n; r++) {
      if (r !== c) fpSum += cm[r][c]; // others predicted as c
      if (r !== c) fnSum += cm[c][r]; // c predicted as others
    }
  }

  const precision = tpSum + fpSum > 0 ? tpSum / (tpSum + fpSum) : 0;
  const recall = tpSum + fnSum > 0 ? tpSum / (tpSum + fnSum) : 0;
  const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
  return { precision, recall, f1 };
}

/**
 * Logarithmic loss (cross-entropy loss).
 *
 * For **binary** classification: yProba is a 1D array of P(class=1).
 * For **multiclass**: yProba is a 2D array [samples x classes], where
 * columns correspond to lexicographically sorted unique classes in yTrue.
 *
 * Probabilities are clipped to [eps, 1-eps] to avoid log(0).
 *
 * @param {Array<*>} yTrue - True labels
 * @param {number[]|number[][]} yProba - Predicted probabilities
 * @returns {number}
 */
export function logLoss(yTrue, yProba) {
  if (!Array.isArray(yTrue) || yTrue.length === 0) {
    throw new Error('logLoss: yTrue must be a non-empty array');
  }
  if (!Array.isArray(yProba) || yProba.length !== yTrue.length) {
    throw new Error('logLoss: yProba must have the same length as yTrue');
  }

  const eps = 1e-15;
  const n = yTrue.length;

  // Binary case: yProba is 1D
  if (!Array.isArray(yProba[0])) {
    const classes = _uniqueClasses(yTrue);
    if (classes.length > 2) {
      throw new Error('logLoss: for multiclass, yProba must be 2D');
    }
    const positiveClass = classes.length === 2 ? classes[1] : classes[0];
    let sum = 0;
    for (let i = 0; i < n; i++) {
      const p = Math.min(Math.max(yProba[i], eps), 1 - eps);
      if (yTrue[i] === positiveClass) {
        sum += Math.log(p);
      } else {
        sum += Math.log(1 - p);
      }
    }
    return -sum / n;
  }

  // Multiclass case: yProba is 2D
  const classes = _uniqueClasses(yTrue);
  const classIdx = new Map(classes.map((c, i) => [c, i]));
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const trueIdx = classIdx.get(yTrue[i]);
    if (trueIdx === undefined) {
      throw new Error(`logLoss: unseen class "${yTrue[i]}" in yTrue`);
    }
    const p = Math.min(Math.max(yProba[i][trueIdx], eps), 1 - eps);
    sum += Math.log(p);
  }
  return -sum / n;
}

/**
 * Area Under the ROC Curve for binary classification (trapezoidal rule).
 *
 * @param {Array<*>} yTrue - Binary labels (two unique values)
 * @param {number[]} yProba - Predicted probabilities for the positive class
 * @returns {number} AUC in [0, 1]
 */
export function rocAucScore(yTrue, yProba) {
  if (!Array.isArray(yTrue) || !Array.isArray(yProba)) {
    throw new Error('rocAucScore: yTrue and yProba must be arrays');
  }
  if (yTrue.length !== yProba.length || yTrue.length === 0) {
    throw new Error('rocAucScore: yTrue and yProba must have the same non-zero length');
  }

  const classes = _uniqueClasses(yTrue);
  if (classes.length !== 2) {
    throw new Error('rocAucScore: only binary classification is supported (found ' + classes.length + ' classes)');
  }

  const positiveClass = classes[1]; // lexicographically larger class is positive

  // Pair probabilities with binary labels
  const pairs = yTrue.map((t, i) => ({
    prob: yProba[i],
    label: t === positiveClass ? 1 : 0,
  }));

  // Sort by descending probability
  pairs.sort((a, b) => b.prob - a.prob);

  const totalPositive = pairs.filter((p) => p.label === 1).length;
  const totalNegative = pairs.length - totalPositive;

  if (totalPositive === 0 || totalNegative === 0) {
    return NaN; // AUC is undefined when only one class present
  }

  // Compute ROC curve points and AUC via trapezoidal rule
  let tpCount = 0;
  let fpCount = 0;
  let prevTPR = 0;
  let prevFPR = 0;
  let auc = 0;

  for (let i = 0; i < pairs.length; i++) {
    if (pairs[i].label === 1) {
      tpCount++;
    } else {
      fpCount++;
    }

    // Only compute when threshold changes or at the last sample
    if (i === pairs.length - 1 || pairs[i].prob !== pairs[i + 1].prob) {
      const tpr = tpCount / totalPositive;
      const fpr = fpCount / totalNegative;
      // Trapezoidal area
      auc += (fpr - prevFPR) * (tpr + prevTPR) / 2;
      prevTPR = tpr;
      prevFPR = fpr;
    }
  }

  return auc;
}

/**
 * Generate a classification report with per-class metrics and averages.
 *
 * @param {Array<*>} yTrue
 * @param {Array<*>} yPred
 * @returns {Object} Report object:
 *   {
 *     perClass: { [label]: { precision, recall, f1, support } },
 *     macroAvg: { precision, recall, f1 },
 *     weightedAvg: { precision, recall, f1 },
 *     accuracy: number,
 *     totalSamples: number
 *   }
 */
export function classificationReport(yTrue, yPred) {
  _validatePair(yTrue, yPred, 'classificationReport');
  const { matrix, labels } = confusionMatrix(yTrue, yPred);

  const perClass = {};
  let macroPrecision = 0;
  let macroRecall = 0;
  let macroF1 = 0;
  let weightedPrecision = 0;
  let weightedRecall = 0;
  let weightedF1 = 0;
  let totalSamples = 0;

  for (let i = 0; i < labels.length; i++) {
    const metrics = _perClassMetrics(matrix, i);
    const support = matrix[i].reduce((s, v) => s + v, 0);
    perClass[labels[i]] = {
      precision: metrics.precision,
      recall: metrics.recall,
      f1: metrics.f1,
      support,
    };
    macroPrecision += metrics.precision;
    macroRecall += metrics.recall;
    macroF1 += metrics.f1;
    weightedPrecision += metrics.precision * support;
    weightedRecall += metrics.recall * support;
    weightedF1 += metrics.f1 * support;
    totalSamples += support;
  }

  const nClasses = labels.length;

  return {
    perClass,
    macroAvg: {
      precision: macroPrecision / nClasses,
      recall: macroRecall / nClasses,
      f1: macroF1 / nClasses,
    },
    weightedAvg: {
      precision: totalSamples > 0 ? weightedPrecision / totalSamples : 0,
      recall: totalSamples > 0 ? weightedRecall / totalSamples : 0,
      f1: totalSamples > 0 ? weightedF1 / totalSamples : 0,
    },
    accuracy: accuracy(yTrue, yPred),
    totalSamples,
  };
}
