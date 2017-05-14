import _ from 'lodash';
import { Pla } from './../';
import trainingData from './../../data/linear_separable/train_data.json';
import testData from './../../data/linear_separable/test_data.json';
import nlTestData from './../../data/not_linear_separable/test_data.json';
import { classificationReport } from './../utils/metrics';

describe('pla', () => {
  let pla;
  const { x, y } = trainingData;
  const processedY = _.map(y, (data) => (data === 0 ? -1 : 1));

  before(() => {
    pla = new Pla();
    pla.training({ x, y: processedY });
  });

  it('predict', () => {
    const yHat = pla.predict(testData.x);

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
