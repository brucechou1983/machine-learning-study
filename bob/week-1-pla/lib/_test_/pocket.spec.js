import _ from 'lodash';
import { Pocket } from './../';
import trainingData from './../../data/not_linear_separable/train_data.json';
import testData from './../../data/not_linear_separable/test_data.json';
import { classificationReport } from './../utils/metrics';

describe('Pocket PLA algorithm', () => {
  let pocket;

  before(() => {
    pocket = new Pocket();
    const { x, y } = trainingData;
    const processedY = _.map(y, (data) => (data === 0 ? -1 : 1));

    pocket.training({ x, y: processedY }, 100);
  });

  it('predict', () => {
    const { x, y } = testData;
    const processedY = _.map(y, (data) => (data === 0 ? -1 : 1));

    const yHat = pocket.predict(x);
    console.table(classificationReport(processedY, yHat));
  });
});
