import _ from 'lodash';

export default function classificationReport(y, yHat) {
  const results = {};
  const checkNewClass = (labels, labelsResult) => {
    _.map(labels, (l) => {
      if (!(l in results)) {
        labelsResult[l] = {
          label: l,
          support: 0,
          tp: 0,
          fp: 0,
          fn: 0,
        };
      }
    });
  };

  // count tp, fp, fn, support (count of each class of y)
  _.map(_.zip(y, yHat), ([truth, prediction]) => {
    // new class
    checkNewClass([truth, prediction], results);

    results[truth].support += 1;

    if (prediction === truth) {
      results[prediction].tp += 1;
    } else {
      results[truth].fn += 1;
      results[prediction].fp += 1;
    }
  });

  /* calculate performance
   precision(P): tp / (tp + fp)
   recall(R): tp / (tp + fn)
   fbeta(F): (1+beta^2)*PR/((beta^2*P)+R) => f1: 2PR/(P+R)
   */
  _.map(results, (result) => {
    result.precision = result.tp / (result.tp + result.fp);
    result.recall = result.tp / (result.tp + result.fn);
    result.f1 = (2 * result.precision * result.recall) / (result.precision + result.recall);
  });

  const report = _.map(results, (result) =>
    _.pick(result, ['label', 'precision', 'recall', 'f1', 'support']));

  return report;
}
