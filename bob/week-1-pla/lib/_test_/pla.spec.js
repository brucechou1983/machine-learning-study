import _ from 'lodash';
import { Pla } from './../';
import trainingData from './../../data/linear_separable/train_data.json';
import testData from './../../data/linear_separable/test_data.json';
import nlTestData from './../../data/not_linear_separable/test_data.json';
import { classificationReport } from './../utils/metrics';

describe('PLA algorithm', () => {
  let pla;

  before(() => {
    pla = new Pla();
    const { x, y } = trainingData;
    const processedY = _.map(y, (data) => (data === 0 ? -1 : 1));

    pla.training({ x, y: processedY });
  });

  it('predict', () => {
    const { x, y } = testData;
    const processedY = _.map(y, (data) => (data === 0 ? -1 : 1));

    const yHat = pla.predict(x);
    console.table(classificationReport(processedY, yHat));
  });
});

describe('pla with the not_linear_separable data', () => {
  it('should throw error', (done) => {
    const pla = new Pla();
    const { x, y } = nlTestData;
    const processedY = _.map(y, (data) => (data === 0 ? -1 : 1));


    assert.throws(() => pla.training({ x, y: processedY }), 'PLA solution cannot be found');

    done();
  });
});
