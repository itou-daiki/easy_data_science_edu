import assert from 'node:assert/strict';

import {
    averagePrecisionScore,
    bootstrapMetricInterval,
    matthewsCorrelationCoefficient,
    meanAbsoluteError,
    multiclassRocAucScore,
    rSquared
} from '../js/ml/metrics.js';
import {
    crossValidateWithPreprocessing,
    GroupKFold,
    learningCurveWithPreprocessing,
    TimeSeriesSplit
} from '../js/ml/model_selection.js';
import { prepareTrainValidationFeatures } from '../js/ml/preprocessing.js';
import { kernelSHAP } from '../js/ml/shap.js';
import { RandomForestClassifier } from '../js/ml/classification/random_forest.js';
import { escapeHtml, toCSV } from '../js/utils.js';
import { buildAnalysisQualityReport } from '../js/analysis_quality.js';
import { createAnnotationState, importAnnotationsJSON } from '../js/analyses/video_analysis/annotation.js';

function assertFiniteMatrix(matrix) {
    assert.ok(matrix.length > 0);
    matrix.forEach(row => row.forEach(value => assert.ok(Number.isFinite(value))));
}

class MeanRegressor {
    fit(X, y) {
        assert.equal(X.length, y.length);
        this.mean = y.reduce((sum, value) => sum + value, 0) / y.length;
        return this;
    }

    predict(X) {
        return X.map(() => this.mean);
    }
}

class AllClassesClassifier {
    static observedClassCounts = [];

    fit(X, y) {
        assert.equal(X.length, y.length);
        const classes = [...new Set(y)];
        AllClassesClassifier.observedClassCounts.push(classes.length);
        assert.equal(classes.length, 3, 'every learning-curve subset must retain all classes');
        this.prediction = classes[0];
        return this;
    }

    predict(X) {
        return X.map(() => this.prediction);
    }
}

const perfectMulticlassAuc = multiclassRocAucScore(
    [0, 1, 2, 0, 1, 2],
    [
        [0.9, 0.05, 0.05],
        [0.05, 0.9, 0.05],
        [0.05, 0.05, 0.9],
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8]
    ],
    [0, 1, 2]
);
assert.equal(perfectMulticlassAuc, 1);
assert.equal(averagePrecisionScore([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8], 1), 1);
assert.equal(averagePrecisionScore([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5], 1), 0.5);
assert.equal(matthewsCorrelationCoefficient([0, 0, 1, 1], [0, 0, 1, 1]), 1);
assert.equal(matthewsCorrelationCoefficient([0, 0, 1, 1], [1, 1, 0, 0]), -1);

const groupFolds = [...new GroupKFold({ nSplits: 3, randomState: 9 }).split(
    Array.from({ length: 12 }, (_, index) => [index]),
    ['a', 'a', 'b', 'b', 'c', 'c', 'd', 'd', 'e', 'e', 'f', 'f']
)];
assert.equal(groupFolds.length, 3);
groupFolds.forEach(([train, test]) => {
    const groups = ['a', 'a', 'b', 'b', 'c', 'c', 'd', 'd', 'e', 'e', 'f', 'f'];
    const trainGroups = new Set(train.map(index => groups[index]));
    assert.ok(test.every(index => !trainGroups.has(groups[index])), 'a group must never cross train/test boundaries');
});

const timeFolds = [...new TimeSeriesSplit({ nSplits: 3, gap: 1 }).split(
    Array.from({ length: 12 }, (_, index) => [index])
)];
assert.equal(timeFolds.length, 3);
timeFolds.forEach(([train, test]) => {
    assert.ok(Math.max(...train) + 1 < Math.min(...test), 'time splits must train strictly before validation with the requested gap');
});

const oneFeatureShap = kernelSHAP(rows => rows.map(([x]) => 2 * x + 1), [[3]], [[0], [2]], {
    nPermutations: 4,
    randomState: 3
});
assert.ok(Math.abs(oneFeatureShap.baseValue + oneFeatureShap.shapValues[0][0] - 7) < 1e-12);
const widePredict = rows => rows.map(row => row.reduce((sum, value, index) => sum + value * (index + 1), 0));
const wideX = [Array.from({ length: 32 }, (_, index) => index / 10)];
const wideBackground = [Array(32).fill(0), Array(32).fill(0.5)];
const wideShapA = kernelSHAP(widePredict, wideX, wideBackground, { nPermutations: 8, randomState: 13 });
const wideShapB = kernelSHAP(widePredict, wideX, wideBackground, { nPermutations: 8, randomState: 13 });
assert.deepEqual(wideShapA, wideShapB, 'SHAP approximation must be reproducible');
assert.equal(wideShapA.shapValues[0].length, 32);
assert.ok(Math.abs(wideShapA.baseValue + wideShapA.shapValues[0].reduce((a, b) => a + b, 0) - widePredict(wideX)[0]) < 1e-10);

assert.ok(Number.isNaN(rSquared([1], [1])), 'R2 is undefined for one observation');
assert.equal(rSquared([3, 3], [3, 3]), 1);
assert.equal(rSquared([3, 3], [2, 2]), 0);

const truth = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
const prediction = [0.2, 0.8, 2.3, 2.9, 4.1, 4.8, 6.2, 6.8, 8.1, 9.2];
const intervalA = bootstrapMetricInterval(truth, prediction, meanAbsoluteError, {
    nResamples: 100,
    randomState: 17
});
const intervalB = bootstrapMetricInterval(truth, prediction, meanAbsoluteError, {
    nResamples: 100,
    randomState: 17
});
assert.deepEqual(intervalA, intervalB, 'bootstrap intervals must be reproducible');
assert.ok(intervalA.lower <= intervalA.estimate && intervalA.estimate <= intervalA.upper);

const forestX = Array.from({ length: 30 }, (_, index) => [index, index % 4]);
const forestY = forestX.map((_, index) => index === 29 ? 2 : 0);
const forestA = new RandomForestClassifier({ nEstimators: 31, maxDepth: 4, randomState: 11 });
const forestB = new RandomForestClassifier({ nEstimators: 31, maxDepth: 4, randomState: 11 });
forestA.fit(forestX, forestY);
forestB.fit(forestX, forestY);
const forestProbaA = forestA.predictProba([[29, 1], [2, 2]]);
const forestProbaB = forestB.predictProba([[29, 1], [2, 2]]);
assert.deepEqual(forestProbaA, forestProbaB, 'random forest must honor randomState');
forestProbaA.forEach(row => {
    assert.equal(row.length, 2);
    row.forEach(value => assert.ok(Number.isFinite(value)));
    assert.ok(Math.abs(row.reduce((sum, value) => sum + value, 0) - 1) < 1e-12);
});

const prepared = prepareTrainValidationFeatures(
    [
        { x: 1, group: 'a', target: 10 },
        { x: 2, group: 'b', target: 12 },
        { x: Number.NaN, group: '', target: 14 },
        { x: Number.POSITIVE_INFINITY, group: null, target: 16 },
        { x: 5, group: 'a', target: 18 },
        { x: 6, group: 'b', target: 20 }
    ],
    [
        { x: Number.NEGATIVE_INFINITY, group: 'a', target: 22 },
        { x: 8, group: 'unseen', target: 24 }
    ],
    'target',
    {
        task: 'regression',
        selectedFeatures: ['x', 'group'],
        scale: false,
        removeOutliers: false,
        removeMulticollinearity: false,
        transformFeatures: false
    }
);
assertFiniteMatrix(prepared.XTrain);
assertFiniteMatrix(prepared.XTest);
assert.ok(prepared.featureNames.some(name => name.startsWith('group=')));
const categoricalIndices = prepared.featureNames
    .map((name, index) => name.startsWith('group=') ? index : -1)
    .filter(index => index >= 0);
assert.ok(categoricalIndices.length >= 2);
assert.ok(
    categoricalIndices.every(index => prepared.XTest[1][index] === 0),
    'an unseen category must be represented by the all-zero One-Hot vector'
);

const regressionRows = Array.from({ length: 24 }, (_, index) => ({
    feature: index,
    target: index * 1.5 + (index % 3)
}));
const cvOptions = {
    cv: 4,
    scoring: 'neg_mae',
    task: 'regression',
    randomState: 23,
    selectedFeatures: ['feature'],
    removeOutliers: false,
    removeMulticollinearity: false,
    transformFeatures: false
};
const cvA = crossValidateWithPreprocessing(MeanRegressor, regressionRows, 'target', cvOptions);
const cvB = crossValidateWithPreprocessing(MeanRegressor, regressionRows, 'target', cvOptions);
assert.equal(cvA.length, 4);
assert.deepEqual(cvA, cvB, 'preprocessing-aware CV must be reproducible');
cvA.forEach(value => assert.ok(Number.isFinite(value)));

const sealedResultReport = buildAnalysisQualityReport({
    data: regressionRows,
    task: 'regression',
    targetCol: 'target',
    selectedFeatures: ['feature'],
    XTrain: regressionRows.slice(0, 18).map(row => [row.feature]),
    XTest: regressionRows.slice(18).map(row => [row.feature]),
    yTrain: regressionRows.slice(0, 18).map(row => row.target),
    yTest: regressionRows.slice(18).map(row => row.target),
    requestedCvFolds: 3,
    effectiveCvFolds: 3,
    result: { cvMean: 0.9, cvStd: 0.02, r2: null, holdoutEvaluated: false }
});
assert.ok(
    sealedResultReport.items.every(item => !item.title.includes('CVとTestの差')),
    'a sealed holdout must not be treated as a zero test score'
);
const revealedResultReport = buildAnalysisQualityReport({
    data: regressionRows,
    task: 'regression',
    targetCol: 'target',
    selectedFeatures: ['feature'],
    XTrain: regressionRows.slice(0, 18).map(row => [row.feature]),
    XTest: regressionRows.slice(18).map(row => [row.feature]),
    yTrain: regressionRows.slice(0, 18).map(row => row.target),
    yTest: regressionRows.slice(18).map(row => row.target),
    requestedCvFolds: 3,
    effectiveCvFolds: 3,
    result: { cvMean: 0.9, cvStd: 0.02, r2: 0.4, holdoutEvaluated: true }
});
assert.ok(revealedResultReport.items.some(item => item.title === 'CVとTestの差が大きい'));

const categoricalEncodingReport = buildAnalysisQualityReport({
    data: [
        { region: 'east', target: 1 },
        { region: 'west', target: 2 },
        { region: 'north', target: 3 }
    ],
    task: 'regression',
    targetCol: 'target',
    selectedFeatures: ['region']
});
assert.ok(
    categoricalEncodingReport.items.some(item => item.title === 'カテゴリ特徴量はOne-Hot Encoding'),
    'the quality report must disclose fold-local One-Hot encoding and unknown-category handling'
);

const classificationRows = Array.from({ length: 45 }, (_, index) => ({
    feature: index,
    target: index % 3
}));
AllClassesClassifier.observedClassCounts = [];
const curve = learningCurveWithPreprocessing(
    AllClassesClassifier,
    {},
    classificationRows,
    'target',
    {
        trainSizes: [0.05, 0.2, 1],
        cv: 3,
        scoring: 'f1',
        stratified: true,
        task: 'classification',
        randomState: 31,
        selectedFeatures: ['feature'],
        removeOutliers: false,
        removeMulticollinearity: false,
        transformFeatures: false
    }
);
assert.equal(curve.trainSizes.length, 3);
assert.ok(AllClassesClassifier.observedClassCounts.every(count => count === 3));

assert.equal(
    escapeHtml('<img src=x onerror="alert(1)">&'),
    '&lt;img src=x onerror=&quot;alert(1)&quot;&gt;&amp;'
);
const csv = toCSV(['name'], [['=2+2'], ['-1'], ['@SUM(A1:A2)'], ['safe']]);
assert.ok(csv.includes('"\t=2+2"'));
assert.ok(csv.includes('"-1"'), 'plain negative numbers must remain numeric text');
assert.ok(csv.includes('"\t@SUM(A1:A2)"'));

const annotationState = createAnnotationState();
importAnnotationsJSON(annotationState, JSON.stringify({
    version: 1,
    items: [{
        id: 'safe-item',
        tool: 'line',
        color: '#ff0000',
        thickness: 4,
        durationSec: 2,
        time: 1.5,
        points: [{ nx: 0.1, ny: 0.2 }, { nx: 0.8, ny: 0.9 }]
    }]
}));
assert.equal(annotationState.items.length, 1);
assert.throws(
    () => importAnnotationsJSON(annotationState, JSON.stringify({
        version: 1,
        items: [{
            tool: 'line', color: 'url(javascript:alert(1))', thickness: 4,
            durationSec: 2, time: 0, points: [{ nx: 0, ny: 0 }]
        }]
    })),
    /色が不正/
);
assert.throws(
    () => importAnnotationsJSON(annotationState, JSON.stringify({
        version: 1,
        items: [{
            tool: 'pen', color: '#ff0000', thickness: 4,
            durationSec: 2, time: 0, points: [{ nx: Number.POSITIVE_INFINITY, ny: 0 }]
        }]
    })),
    /x座標が範囲外/
);

console.log('ml_reliability.test.mjs: all assertions passed');
