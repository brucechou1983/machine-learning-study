import _ from 'lodash';
import Debug from 'debug';
import Pla, {
  _initWeights,
  _updateWeights,
  _createSampleVectors,
  _maxIterator,
  _weight,
} from './../pla';

const _pla = Symbol();
const _countError = Symbol();
const _bestModel = Symbol();
const _debugTraining = Debug('Pocket:training');
const _debugPredict = Debug('PLA:predict');

export default class Pocket {

  constructor(weight, maxIterator) {
    this[_pla] = new Pla(weight, maxIterator);
    this[_maxIterator] = this[_pla][_maxIterator];
  }

  get [_createSampleVectors]() {
    return this[_pla][_createSampleVectors];
  }

  get [_initWeights]() {
    return this[_pla][_initWeights];
  }

  get [_updateWeights]() {
    return this[_pla][_updateWeights];
  }

  get [_countError]() {
    return (samples, labels, weight, sampleCount) => {
      let errorCounts = 0;

      if (sampleCount <= 0) {
        _.forEach(samples, (sample, index) => {
          const label = labels[index];
          const signOfWx = Math.sign(sample.dot(weight));
          errorCounts += signOfWx !== label ? 1 : 0;
        });

        return errorCounts;
      }

      // random sample to speed up pocket
      _.map(_.sampleSize(_.zip(samples, labels), sampleCount), ([sample, label]) => {
        const signOfWx = Math.sign(sample.dot(weight));
        errorCounts += signOfWx !== label ? 1 : 0;
      });

      return errorCounts;
    };
  }

  predict(sample) {
    if (_.isNil(this[_bestModel])) {
      throw new Error('You need to train the model first!');
    }

    const { errorCounts, weight } = this[_bestModel];
    _debugPredict(`errorCounts: ${errorCounts}`);
    _debugPredict(`weight: ${weight}`);

    this[_pla][_weight] = weight;

    return this[_pla].predict(sample);
  }

  training(data, sampleCount = 0) {
    let done = false;
    let iterator = 0;
    const { x, y } = data;

    let weight = this[_pla][_weight] || this[_initWeights](x[0].length);
    this[_pla][_weight] = weight;
    const xVectors = _.map(x, (features) => this[_createSampleVectors](features));

    while (!done) {
      if (iterator === this[_maxIterator]) {
        break;
      }

      done = this[_updateWeights](xVectors, y);
      weight = this[_pla].weight;

      iterator += 1;

      const errorCounts = this[_countError](xVectors, y, weight, sampleCount);
      if (!this[_bestModel] || (this[_bestModel].errorCounts > errorCounts)) {
        this[_bestModel] = { errorCounts, weight };
      }

      _debugTraining(`iterator: ${iterator}`);
    }
  }

  get weight() {
    return this[_bestModel].weight;
  }
}
