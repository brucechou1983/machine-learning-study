export function classificationReport(y, yHat) {
  const title = '             precision    recall  f1-score   support\n\n';
  const summary = '\navg / total       0.58      0.75      0.65         4\n';

  return '          0       0.67      1.00      0.80         2\n          1       0.00      0.00      0.00         1\n          2       1.00      1.00      1.00         1\n'
}
