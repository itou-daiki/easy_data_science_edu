// ==========================================
// Analysis quality checks for educational AutoML
// ==========================================
import { formatNumber } from './utils.js';

const LEVEL_META = {
    danger: { label: '要確認', icon: 'fa-triangle-exclamation' },
    warning: { label: '注意', icon: 'fa-circle-exclamation' },
    info: { label: '参考', icon: 'fa-circle-info' },
    good: { label: '良好', icon: 'fa-circle-check' }
};

/**
 * Build a data and evaluation reliability report for table-based ML.
 * @param {Object} options
 * @returns {{ overall: string, counts: Object, items: Object[] }}
 */
export function buildAnalysisQualityReport(options = {}) {
    const {
        data = [],
        task = 'regression',
        targetCol = '',
        selectedFeatures = [],
        requestedCvFolds = null,
        effectiveCvFolds = null,
        XTrain = [],
        XTest = [],
        yTrain = [],
        yTest = [],
        preprocessInfo = null,
        result = null
    } = options;

    const items = [];
    const add = (level, title, detail) => {
        items.push({ level, title, detail });
    };

    const rowCount = Array.isArray(data) ? data.length : 0;
    const trainRows = Array.isArray(XTrain) ? XTrain.length : 0;
    const testRows = Array.isArray(XTest) ? XTest.length : 0;
    const featureCount = selectedFeatures.length;
    const targetValues = getColumnValues(data, targetCol);

    checkSampleSize(add, rowCount, trainRows, testRows);
    checkFeatureVolume(add, featureCount, trainRows);
    checkMissingValues(add, data, targetCol, selectedFeatures);
    checkTarget(add, task, targetValues, yTrain, targetCol);
    checkFeatureRisks(add, data, task, targetCol, selectedFeatures, targetValues);
    checkCvSettings(add, requestedCvFolds, effectiveCvFolds);
    checkPreprocessing(add, rowCount, preprocessInfo);
    checkResult(add, task, result);

    add(
        'warning',
        'CVは前処理後データでの参考値',
        '現在の比較処理では、欠損補完・変換・標準化などを済ませた訓練データに対してCVを行います。独立テスト指標を主に確認してください。'
    );

    const counts = items.reduce((acc, item) => {
        acc[item.level] = (acc[item.level] || 0) + 1;
        return acc;
    }, { danger: 0, warning: 0, info: 0, good: 0 });

    const overall = counts.danger > 0 ? 'danger' : counts.warning > 0 ? 'warning' : 'good';
    return { overall, counts, items };
}

/**
 * Render a reliability report as an HTML panel.
 * @param {{ overall: string, counts: Object, items: Object[] }} report
 * @param {Object} options
 * @returns {string}
 */
export function renderAnalysisQualityPanel(report, options = {}) {
    if (!report) return '';
    const title = options.title || '分析信頼性チェック';
    const maxItems = options.maxItems || 12;
    const visibleItems = report.items.slice(0, maxItems);
    const hiddenCount = Math.max(0, report.items.length - visibleItems.length);
    const statusText = report.overall === 'danger'
        ? '重要な確認点があります'
        : report.overall === 'warning'
            ? '注意点があります'
            : '大きな問題は見つかりません';

    return `
        <div class="analysis-quality-panel analysis-quality-${report.overall}">
            <div class="analysis-quality-header">
                <div>
                    <h4><i class="fas fa-shield-halved"></i> ${escapeHtml(title)}</h4>
                    <p>${escapeHtml(statusText)}。結果を読む前に、データと評価条件を確認します。</p>
                </div>
                <div class="analysis-quality-badges">
                    ${renderCountBadge('danger', report.counts.danger)}
                    ${renderCountBadge('warning', report.counts.warning)}
                    ${renderCountBadge('info', report.counts.info)}
                    ${renderCountBadge('good', report.counts.good)}
                </div>
            </div>
            <div class="analysis-quality-items">
                ${visibleItems.map(renderQualityItem).join('')}
                ${hiddenCount > 0 ? `<div class="analysis-quality-more">ほか ${hiddenCount} 件の確認項目があります。</div>` : ''}
            </div>
        </div>
    `;
}

/**
 * Extract important warning notes for AI assistant context.
 * @param {{ items: Object[] }} report
 * @param {number} limit
 * @returns {string[]}
 */
export function getAnalysisQualityNotes(report, limit = 6) {
    if (!report?.items) return [];
    return report.items
        .filter(item => item.level === 'danger' || item.level === 'warning')
        .slice(0, limit)
        .map(item => `${item.title}: ${item.detail}`);
}

function checkSampleSize(add, rowCount, trainRows, testRows) {
    if (rowCount < 30) {
        add('danger', 'サンプル数が少ない', `${rowCount}行です。機械学習の評価は大きく揺れやすいため、学習用の実験結果として扱ってください。`);
    } else if (rowCount < 100) {
        add('warning', 'サンプル数がやや少ない', `${rowCount}行です。テスト分割やCVのfoldによって順位が変わる可能性があります。`);
    } else {
        add('good', 'サンプル数', `${rowCount}行あり、学習用デモとしては比較しやすい規模です。`);
    }

    if (testRows > 0 && testRows < 10) {
        add('warning', 'テスト件数が少ない', `テストデータが${testRows}件です。独立評価の1件あたりの影響が大きくなります。`);
    }
    if (trainRows > 0 && trainRows < 20) {
        add('warning', '訓練件数が少ない', `訓練データが${trainRows}件です。複雑なモデルでは過学習に注意してください。`);
    }
}

function checkFeatureVolume(add, featureCount, trainRows) {
    if (featureCount === 0) {
        add('danger', '特徴量がありません', 'モデルを学習するには少なくとも1つの特徴量が必要です。');
        return;
    }
    if (trainRows > 0 && featureCount >= trainRows / 3) {
        add('danger', '特徴量が多すぎる可能性', `訓練${trainRows}件に対して特徴量${featureCount}個です。偶然の当たりや過学習が起きやすい状態です。`);
    } else if (trainRows > 0 && featureCount >= trainRows / 10) {
        add('warning', '特徴量数に注意', `訓練${trainRows}件に対して特徴量${featureCount}個です。モデル解釈と過学習を確認してください。`);
    } else {
        add('good', '特徴量数', `特徴量${featureCount}個です。データ件数に対して極端に多い状態ではありません。`);
    }
}

function checkMissingValues(add, data, targetCol, selectedFeatures) {
    const columns = [targetCol, ...selectedFeatures].filter(Boolean);
    const rowCount = data.length || 1;
    const summaries = columns.map(col => {
        const missing = data.filter(row => isMissing(row?.[col])).length;
        return { col, missing, rate: missing / rowCount };
    });
    const highMissing = summaries.filter(item => item.rate >= 0.5);
    const midMissing = summaries.filter(item => item.rate >= 0.2 && item.rate < 0.5);
    const anyMissing = summaries.filter(item => item.missing > 0);

    if (highMissing.length > 0) {
        add('danger', '欠損率が高い列があります', `${formatColumnRates(highMissing)}。補完後の値に分析結果が引っ張られる可能性があります。`);
    } else if (midMissing.length > 0) {
        add('warning', '欠損率に注意', `${formatColumnRates(midMissing)}。欠損理由を確認すると解釈しやすくなります。`);
    } else if (anyMissing.length > 0) {
        add('info', '欠損値があります', `${formatColumnRates(anyMissing)}。自動補完後の評価であることを踏まえて読んでください。`);
    } else {
        add('good', '欠損値', '選択した目的変数・特徴量には欠損が見つかりません。');
    }
}

function checkTarget(add, task, targetValues, yTrain, targetCol) {
    const validTargets = targetValues.filter(value => !isMissing(value));
    if (validTargets.length === 0) {
        add('danger', '目的変数が空です', `${targetCol} に有効な値がありません。`);
        return;
    }

    if (task === 'regression') {
        const numeric = validTargets.map(Number).filter(Number.isFinite);
        const unique = new Set(numeric).size;
        const std = standardDeviation(numeric);
        if (numeric.length !== validTargets.length) {
            add('danger', '回帰の目的変数に非数値があります', '回帰では目的変数が連続的な数値である必要があります。');
        } else if (std === 0 || unique <= 1) {
            add('danger', '目的変数に変動がありません', '予測対象が一定のため、回帰モデルの性能評価が意味を持ちにくい状態です。');
        } else if (unique < 10) {
            add('warning', '目的変数の値の種類が少ない', `目的変数のユニーク値が${unique}種類です。分類問題として扱う方が自然な場合があります。`);
        } else {
            add('good', '目的変数', '回帰対象として使える数値のばらつきがあります。');
        }
        return;
    }

    const counts = countValues(validTargets);
    const classCount = counts.length;
    const majority = counts[0];
    const minClass = counts[counts.length - 1];
    const majorityRate = majority.count / validTargets.length;
    if (classCount < 2) {
        add('danger', '分類クラスが1種類だけです', '分類では目的変数に2種類以上のクラスが必要です。');
    } else if (majorityRate >= 0.95) {
        add('danger', 'クラス不均衡が非常に大きい', `最多クラス「${majority.label}」が${formatPercent(majorityRate)}を占めています。Accuracyは過信できません。`);
    } else if (majorityRate >= 0.8) {
        add('warning', 'クラス不均衡に注意', `最多クラス「${majority.label}」が${formatPercent(majorityRate)}を占めています。Macro F1とRecallを重視してください。`);
    } else {
        add('good', 'クラス分布', `${classCount}クラスで、極端な多数派偏りは見つかりません。`);
    }
    if (minClass && minClass.count < 5) {
        add('warning', '少数クラスの件数が少ない', `最少クラス「${minClass.label}」は${minClass.count}件です。層化分割後の評価が不安定になります。`);
    }
    if (Array.isArray(yTrain) && yTrain.length > 0) {
        const encodedCounts = countValues(yTrain);
        const trainMin = encodedCounts[encodedCounts.length - 1];
        if (trainMin && trainMin.count < 3) {
            add('warning', '訓練データ内の少数クラスが少ない', `訓練分割後の最少クラスは${trainMin.count}件です。CVのfold数を少なくする方が安定します。`);
        }
    }
}

function checkFeatureRisks(add, data, task, targetCol, selectedFeatures, targetValues) {
    const targetNorm = normalizeName(targetCol);
    const nameRisks = selectedFeatures.filter(feature => {
        const featureNorm = normalizeName(feature);
        if (featureNorm === targetNorm) return false;
        return (featureNorm.length >= 3 && targetNorm.includes(featureNorm))
            || (targetNorm.length >= 3 && featureNorm.includes(targetNorm))
            || /(^id$|id$|_id$|番号|コード|連番|no$)/i.test(feature);
    });
    if (nameRisks.length > 0) {
        add('warning', 'リークまたはID列の候補', `${nameRisks.slice(0, 4).join(', ')} は目的変数に近い名前、またはID/コード系の名前です。予測時にも使える列か確認してください。`);
    }

    const constantFeatures = selectedFeatures.filter(feature => {
        const values = getColumnValues(data, feature).filter(value => !isMissing(value));
        return new Set(values.map(String)).size <= 1;
    });
    if (constantFeatures.length > 0) {
        add('warning', '変化しない特徴量があります', `${constantFeatures.slice(0, 4).join(', ')} は値が一定です。予測にはほぼ寄与しません。`);
    }

    if (task === 'regression') {
        const leakageCandidates = selectedFeatures
            .map(feature => {
                const paired = pairNumericColumns(data, feature, targetCol);
                return { feature, corr: paired.length >= 5 ? pearson(paired.map(p => p.x), paired.map(p => p.y)) : 0 };
            })
            .filter(item => Math.abs(item.corr) >= 0.95)
            .sort((a, b) => Math.abs(b.corr) - Math.abs(a.corr));

        if (leakageCandidates.some(item => Math.abs(item.corr) >= 0.98)) {
            add('danger', '目的変数に近すぎる特徴量があります', `${formatCorrelationList(leakageCandidates)}。集計後の答えや派生列が混ざっていないか確認してください。`);
        } else if (leakageCandidates.length > 0) {
            add('warning', '強い相関の特徴量があります', `${formatCorrelationList(leakageCandidates)}。妥当な説明変数か、目的変数から作った列でないか確認してください。`);
        }
    } else {
        const exactMatches = selectedFeatures
            .map(feature => {
                const rate = exactMatchRate(getColumnValues(data, feature), targetValues);
                return { feature, rate };
            })
            .filter(item => item.rate >= 0.9)
            .sort((a, b) => b.rate - a.rate);
        if (exactMatches.length > 0) {
            add('danger', '目的変数とほぼ同じ特徴量があります', `${exactMatches.slice(0, 3).map(item => `${item.feature} (${formatPercent(item.rate)})`).join(', ')}。答えそのものが特徴量に含まれていないか確認してください。`);
        }
    }
}

function checkCvSettings(add, requestedCvFolds, effectiveCvFolds) {
    if (requestedCvFolds == null || effectiveCvFolds == null) return;
    if (effectiveCvFolds < requestedCvFolds) {
        add('warning', 'CV Fold数を自動調整しました', `指定は${requestedCvFolds}-Foldですが、データ件数に合わせて${effectiveCvFolds}-Foldで実行します。`);
    } else if (effectiveCvFolds < 5) {
        add('info', 'CV Fold数', `${effectiveCvFolds}-Foldです。小規模データでは安定性と計算量のバランスを見て調整してください。`);
    } else {
        add('good', 'CV Fold数', `${effectiveCvFolds}-Foldで比較します。`);
    }
}

function checkPreprocessing(add, rowCount, preprocessInfo) {
    if (!preprocessInfo) return;
    const outlierRows = preprocessInfo.outlierRows || 0;
    const outlierRate = rowCount > 0 ? outlierRows / rowCount : 0;
    if (outlierRate >= 0.15) {
        add('warning', '外れ値除去の影響が大きい', `${outlierRows}行がIQR法で除去されています。外れ値が誤入力か重要な例外か確認してください。`);
    } else if (outlierRows > 0) {
        add('info', '外れ値除去', `${outlierRows}行がIQR法で除去されています。`);
    }

    const removedMulti = preprocessInfo.removedMulticollinear || [];
    if (removedMulti.length > 0) {
        add('info', '多重共線性の自動除去', `${removedMulti.slice(0, 4).join(', ')} を除去しています。係数解釈では残った列だけを見てください。`);
    }
}

function checkResult(add, task, result) {
    if (!result) return;
    const cvStd = result.cvStd;
    if (Number.isFinite(cvStd) && cvStd >= (task === 'regression' ? 0.2 : 0.12)) {
        add('warning', 'CVスコアのばらつきが大きい', `CV標準偏差が${formatNumber(cvStd)}です。foldごとのデータ差に結果が左右されています。`);
    }

    if (task === 'regression') {
        const gap = result.cvMean - result.r2;
        if (Number.isFinite(gap) && gap >= 0.3) {
            add('danger', 'CVとTestの差が大きい', `CV R²がTest R²より${formatNumber(gap)}高いです。過学習、データ分割差、前処理CVの過大評価を疑ってください。`);
        } else if (Number.isFinite(gap) && gap >= 0.15) {
            add('warning', 'CVとTestの差に注意', `CV R²がTest R²より${formatNumber(gap)}高いです。独立テスト指標を優先してください。`);
        }
        if (Number.isFinite(result.r2) && result.r2 < 0) {
            add('warning', 'Test R²が負です', 'テストデータでは平均値予測より悪い可能性があります。特徴量・外れ値・目的変数を見直してください。');
        }
        if (result.baseline && Number.isFinite(result.rmse) && Number.isFinite(result.baseline.rmse)) {
            if (result.rmse >= result.baseline.rmse || result.mae >= result.baseline.mae) {
                add('warning', '単純ベースラインに十分勝っていません', '訓練平均を予測するだけの方法と比べて誤差改善が小さいため、モデルの有用性を慎重に見てください。');
            } else {
                add('good', 'ベースライン比較', '訓練平均ベースラインより誤差が小さくなっています。');
            }
        }
        return;
    }

    const gap = result.cvMean - result.f1;
    if (Number.isFinite(gap) && gap >= 0.2) {
        add('danger', 'CVとTestの差が大きい', `CV F1がTest F1より${formatNumber(gap)}高いです。過学習や分割差を疑ってください。`);
    } else if (Number.isFinite(gap) && gap >= 0.1) {
        add('warning', 'CVとTestの差に注意', `CV F1がTest F1より${formatNumber(gap)}高いです。混同行列とクラス別Recallを重視してください。`);
    }
    if (result.baseline && Number.isFinite(result.f1) && Number.isFinite(result.baseline.f1)) {
        if (result.f1 <= result.baseline.f1 + 0.02) {
            add('warning', '多数派ベースラインに十分勝っていません', '多数派クラスだけを予測する方法とMacro F1が近いため、特徴量の情報量を見直してください。');
        } else {
            add('good', 'ベースライン比較', '多数派クラスベースラインよりMacro F1が改善しています。');
        }
    }
}

function renderCountBadge(level, count) {
    if (!count) return '';
    return `<span class="analysis-quality-count analysis-quality-count-${level}">${LEVEL_META[level].label} ${count}</span>`;
}

function renderQualityItem(item) {
    const meta = LEVEL_META[item.level] || LEVEL_META.info;
    return `
        <div class="analysis-quality-item analysis-quality-item-${item.level}">
            <i class="fas ${meta.icon}"></i>
            <div>
                <strong>${escapeHtml(item.title)}</strong>
                <p>${escapeHtml(item.detail)}</p>
            </div>
        </div>
    `;
}

function getColumnValues(data, col) {
    if (!Array.isArray(data) || !col) return [];
    return data.map(row => row?.[col]);
}

function isMissing(value) {
    return value == null || Number.isNaN(value) || (typeof value === 'string' && value.trim() === '');
}

function standardDeviation(values) {
    if (values.length <= 1) return 0;
    const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
    const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (values.length - 1);
    return Math.sqrt(variance);
}

function countValues(values) {
    const counts = new Map();
    values.forEach(value => {
        const label = String(value);
        counts.set(label, (counts.get(label) || 0) + 1);
    });
    return [...counts.entries()]
        .map(([label, count]) => ({ label, count }))
        .sort((a, b) => b.count - a.count);
}

function normalizeName(value) {
    return String(value || '').toLowerCase().replace(/[\s_\-()（）［\][\].]/g, '');
}

function pairNumericColumns(data, feature, target) {
    return data
        .map(row => ({ x: Number(row?.[feature]), y: Number(row?.[target]) }))
        .filter(pair => Number.isFinite(pair.x) && Number.isFinite(pair.y));
}

function pearson(x, y) {
    if (x.length !== y.length || x.length < 2) return 0;
    const meanX = x.reduce((sum, value) => sum + value, 0) / x.length;
    const meanY = y.reduce((sum, value) => sum + value, 0) / y.length;
    let num = 0;
    let denX = 0;
    let denY = 0;
    for (let i = 0; i < x.length; i++) {
        const dx = x[i] - meanX;
        const dy = y[i] - meanY;
        num += dx * dy;
        denX += dx * dx;
        denY += dy * dy;
    }
    const den = Math.sqrt(denX * denY);
    return den === 0 ? 0 : num / den;
}

function exactMatchRate(values, targetValues) {
    const n = Math.min(values.length, targetValues.length);
    if (n === 0) return 0;
    let valid = 0;
    let matches = 0;
    for (let i = 0; i < n; i++) {
        if (isMissing(values[i]) || isMissing(targetValues[i])) continue;
        valid++;
        if (String(values[i]) === String(targetValues[i])) matches++;
    }
    return valid === 0 ? 0 : matches / valid;
}

function formatColumnRates(items) {
    return items
        .slice(0, 4)
        .map(item => `${item.col}: ${item.missing}件 (${formatPercent(item.rate)})`)
        .join(', ');
}

function formatCorrelationList(items) {
    return items
        .slice(0, 3)
        .map(item => `${item.feature}: r=${formatNumber(item.corr)}`)
        .join(', ');
}

function formatPercent(value) {
    if (!Number.isFinite(value)) return '-';
    return `${(value * 100).toFixed(1)}%`;
}

function escapeHtml(value) {
    return String(value ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}
