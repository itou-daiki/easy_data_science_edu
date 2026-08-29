import assert from 'node:assert/strict';

import { deserializeModel, serializeModel } from '../js/utils.js';
import { reconstructModel } from '../js/analyses/prediction_mode.js';
import { LinearRegression } from '../js/ml/regression/linear.js';
import { RidgeRegression } from '../js/ml/regression/ridge.js';
import { LassoRegression } from '../js/ml/regression/lasso.js';
import { DecisionTreeRegressor } from '../js/ml/regression/decision_tree.js';
import { RandomForestRegressor } from '../js/ml/regression/random_forest.js';
import { KNNRegressor } from '../js/ml/regression/knn.js';
import { GradientBoostingRegressor } from '../js/ml/regression/gradient_boosting.js';
import { LogisticRegression } from '../js/ml/classification/logistic.js';
import { DecisionTreeClassifier } from '../js/ml/classification/decision_tree.js';
import { RandomForestClassifier } from '../js/ml/classification/random_forest.js';
import { KNNClassifier } from '../js/ml/classification/knn.js';
import { GaussianNaiveBayes } from '../js/ml/classification/naive_bayes.js';
import { SVMClassifier } from '../js/ml/classification/svm.js';
import { GradientBoostingClassifier } from '../js/ml/classification/gradient_boosting.js';

function assertNestedClose(actual, expected, tolerance = 1e-12) {
    assert.equal(actual.length, expected.length);
    actual.forEach((value, index) => {
        if (Array.isArray(value)) assertNestedClose(value, expected[index], tolerance);
        else assert.ok(Math.abs(value - expected[index]) <= tolerance, `${value} != ${expected[index]}`);
    });
}

function roundTrip(model, badge, taskType, X, y, fit = true) {
    if (fit) model.fit(X, y);
    const exported = serializeModel(
        { model, name: badge, badge },
        {
            featureNames: ['x1', 'x2'],
            inputFeatureNames: ['x1', 'x2'],
            targetCol: 'target',
            fileName: 'roundtrip.csv',
            taskType,
            classLabels: taskType === 'classification' ? model.classes : null
        }
    );
    const jsonRoundTrip = deserializeModel(JSON.parse(JSON.stringify(exported)));
    const restored = reconstructModel(jsonRoundTrip.modelInfo, taskType);
    assert.deepEqual(restored.predict(X), model.predict(X), `${taskType}/${badge} predictions changed after export`);
    if (taskType === 'classification' && typeof model.predictProba === 'function') {
        assertNestedClose(restored.predictProba(X), model.predictProba(X));
    }
}

const regressionX = Array.from({ length: 18 }, (_, i) => [i / 3, (i % 5) - 2]);
const regressionY = regressionX.map(([x1, x2]) => 3 + 1.5 * x1 - 0.8 * x2 + (x1 % 2) * 0.1);
const fittedLinear = new LinearRegression();
fittedLinear.coefficients = [1.5, -0.8];
fittedLinear.intercept = 3;
fittedLinear.nFeatures = 2;
const fittedRidge = new RidgeRegression({ alpha: 0.5 });
fittedRidge.coefficients = [1.4, -0.7];
fittedRidge.intercept = 2.9;
fittedRidge.nFeatures = 2;
const fittedLasso = new LassoRegression({ alpha: 0.01, maxIter: 200 });
fittedLasso.coefficients = [1.45, -0.75];
fittedLasso.intercept = 3.1;
fittedLasso.nFeatures = 2;
const regressionModels = [
    [fittedLinear, 'Linear', false],
    [fittedRidge, 'Ridge', false],
    [fittedLasso, 'Lasso', false],
    [new DecisionTreeRegressor({ maxDepth: 3 }), 'Tree'],
    [new RandomForestRegressor({ nEstimators: 7, maxDepth: 3, randomState: 7 }), 'RF'],
    [new KNNRegressor({ nNeighbors: 3, weights: 'distance' }), 'KNN'],
    [new GradientBoostingRegressor({ nEstimators: 7, maxDepth: 2, randomState: 7 }), 'GBM']
];
regressionModels.forEach(([model, badge, fit = true]) => roundTrip(model, badge, 'regression', regressionX, regressionY, fit));

const classificationX = Array.from({ length: 24 }, (_, i) => [i % 8, Math.floor(i / 8)]);
const classificationY = classificationX.map(([x1, x2]) => (x1 + x2 * 2) % 3);
const classificationModels = [
    [new LogisticRegression({ maxIter: 150, learningRate: 0.03 }), 'LR'],
    [new DecisionTreeClassifier({ maxDepth: 4 }), 'Tree'],
    [new RandomForestClassifier({ nEstimators: 7, maxDepth: 4, randomState: 7 }), 'RF'],
    [new KNNClassifier({ nNeighbors: 3, weights: 'distance' }), 'KNN'],
    [new GaussianNaiveBayes(), 'NB'],
    [new SVMClassifier({ maxIter: 200, randomState: 7 }), 'SVM'],
    [new GradientBoostingClassifier({ nEstimators: 7, maxDepth: 2, randomState: 7 }), 'GBM']
];
classificationModels.forEach(([model, badge]) => roundTrip(model, badge, 'classification', classificationX, classificationY));

const invalidLinear = serializeModel(
    { model: fittedLinear, name: 'Linear', badge: 'Linear' },
    { featureNames: ['x1', 'x2'], targetCol: 'target', taskType: 'regression' }
);
invalidLinear.modelInfo.params.coefficients = [1];
assert.throws(() => deserializeModel(invalidLinear), /係数の次元/);

console.log('model_roundtrip.test.mjs: all assertions passed');
