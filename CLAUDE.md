# easyDataScience

---
description:
alwaysApply: true
---

## Communication
- **Language**: Always respond, explain your thoughts, and write commit messages/summaries in Japanese, even though these instructions are in English.

---

## Workflow Orchestration

### 1. Plan Mode Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately – don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

### 2. Subagent Strategy
- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution

### 3. Self-Improvement Loop
- After ANY correction from the user: update tasks/lessons.md with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

### 4. Verification Before Done
- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

### 5. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes – don't over-engineer
- Challenge your own work before presenting it

### 6. Autonomous Bug Fixing
- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests – then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

---

## Task Management

1. **Plan First**: Write plan to tasks/todo.md with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review section to tasks/todo.md
6. **Capture Lessons**: Update tasks/lessons.md after corrections

---

## Core Principles

- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimal Impact**: Changes should only touch what's necessary. Avoid introducing bugs.


## Project Overview

PyCaret-like AutoML web application running entirely in the browser. All ML algorithms are implemented from scratch in pure JavaScript - no Python, no backend, no external ML libraries. Educational tool for Japanese learners.

## Architecture

```
index.html                     # Single entry point (CDN libs + ES6 module)
css/style.css                  # All styles, CSS variables for theming
js/
  main.js                      # App orchestrator: file upload, routing, data analysis
  utils.js                     # UI helpers: plots, tables, step indicators, formatNumber
  analyses/                    # UI modules (PyCaret workflow)
    eda.js                     #   Exploratory Data Analysis (distribution/correlation/missing)
    preprocessing.js           #   Data preprocessing UI (scaling/encoding/outliers)
    regression.js              #   Regression AutoML: setup → compare(CV) → tune → predict
    classification.js          #   Classification AutoML: setup → compare(CV) → tune → predict
  ml/                          # Pure ML implementations
    metrics.js                 #   R², MAE, MSE, RMSE, accuracy, F1, precision, recall, AUC, logLoss
    preprocessing.js           #   StandardScaler, MinMaxScaler, LabelEncoder, prepareFeatures()
    model_selection.js         #   trainTestSplit, KFold, StratifiedKFold, crossValidate, gridSearch
    regression/                #   7 regressors: linear, ridge, lasso, decision_tree, random_forest, knn, gradient_boosting
    classification/            #   7 classifiers: logistic, decision_tree, random_forest, knn, naive_bayes, svm, gradient_boosting
datasets/
  regression_demo.csv          # Housing price (150 rows, 7 cols)
  classification_demo.csv      # Customer churn (200 rows, 7 cols)
```

## Tech Stack

- **No build system** - runs directly in browser, no npm/webpack/vite
- **ES6 modules** with dynamic import for analysis modules
- **CDN dependencies**: Plotly.js (charts), math.js (linear algebra), jStat (statistics), XLSX/SheetJS (file parsing), Font Awesome (icons)
- **Dev server**: `python3 -m http.server 8765` from project root

## Key Patterns

- **ML model interface**: All models implement `constructor(params)`, `fit(X, y)`, `predict(X)`, `getParams()`, optionally `predictProba(X)` and `getFeatureImportance()`
- **Immutability**: preprocessing and model_selection functions never mutate input arrays
- **Module-level `_state`** in regression.js/classification.js shares data between PyCaret steps (compare → tune → interpret → predict)
- **prepareFeatures()** returns `{ X, y, featureNames, encoders, scaler, labelEncoder }` - scaler needed for predict_model
- **_cloneModel()** uses `getParams()` method for CV/GridSearch model cloning
- **permutationImportance()** model-agnostic feature importance via feature shuffling
- **learningCurve()** train/test scores at varying training sizes for overfit/underfit diagnosis

## Conventions

- UI text is Japanese (日本語)
- Color scheme: regression = #d97706 (amber), classification = #0891b2 (cyan)
- File sizes: ML models ~80-150 lines, analysis modules ~300-430 lines
- Private helpers prefixed with underscore: `_mean()`, `_std()`, `_cloneModel()`
- JSDoc comments on all public APIs in ml/ directory

## Development Workflow

1. Edit JS files directly
2. Refresh browser at `http://localhost:8765`
3. Verify with browser DevTools console (target: 0 errors excluding favicon)
4. Use Playwright MCP for automated browser testing

## Testing

- No formal test framework (educational project)
- Manual verification via Playwright browser automation
- Console error count = 0 (favicon 404 excluded) as quality gate
- Demo datasets for regression and classification smoke testing

## Common Tasks

### Adding a new regression model
1. Create `js/ml/regression/your_model.js` with `fit(X, y)`, `predict(X)`, `getParams()`, `getFeatureImportance()`
2. Import and add to `MODELS` array in `js/analyses/regression.js`
3. Optionally add to `PARAM_GRIDS` for tune_model support

### Adding a new classification model
Same pattern as regression, but in `js/ml/classification/` and `js/analyses/classification.js`. Also implement `predictProba(X)` for AUC/ROC support.

### Modifying the PyCaret workflow
- Setup UI: `render()` function in regression.js/classification.js
- Compare: `runComparison()` - trains models, runs crossValidate()
- Tune: `runTuneModel()` - calls gridSearch() with PARAM_GRIDS
- Interpret: `runInterpretModel()` - Permutation Importance, PDP, Learning Curve + 総合分析
- Predict: `runPredictModel()` - applies scaler.transform() then model.predict()
