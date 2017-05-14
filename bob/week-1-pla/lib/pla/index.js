import _ from 'lodash';
import { Vector } from 'vectorious';
import Debug from 'debug';

const _initWeights = Symbol();
const _updateWeights = Symbol();
const _createSampleVectors = Symbol();
const _weight = Symbol();
const _maxIterator = 10000;
const _debugUpdate = Debug('PLA:update');
const _debugTraining = Debug('PLA:training');

// sign(w0 * 1 + w1x1 + w2x2)
export default class Pla {

  constructor(weight) {
    this[_weight] = weight;
  }

  get [_createSampleVectors]() {
    // 1 is for w0
    return (features) => new Vector(_.concat([1.0], features));
  }

  get [_initWeights]() {
    // 1 is for w0
    return (dim) => Vector.zeros(dim + 1);
  }

  get [_updateWeights]() {
    return (samples, labels) => {
      let done = true;

      _.forEach(samples, (sample, index) => {
        const label = labels[index];
        const signOfWx = Math.sign(sample.dot(this[_weight]));
        if (signOfWx !== label) {
          _debugUpdate('index-sample, weight, label');
          _debugUpdate(`error case: ${index}-${sample}, ${this[_weight]}, ${label}`);

          // update weight
          const fixer = label > 0 ? 'add' : 'subtract';
          this[_weight] = this[_weight][fixer](sample);

          _debugUpdate(`new weights: ${this[_weight]}`);
          _debugUpdate('*************************************');

          done = false;

          // early return
          return false;
        }

        return true;
      });

      return done;
    };
  }

  predict(sample) {
    if (_.isNil(this[_weight])) {
      throw new Error('You need to train the model first!');
    }

    const xVectors = _.map(sample, (features) => this[_createSampleVectors](features));

    return _.map(xVectors, (xVector) => Math.sign(xVector.dot(this[_weight])));
  }

  training(data) {
    let done = false;
    let iterator = 0;
    const { x, y } = data;

    this[_weight] = this[_initWeights](x[0].length);
    const xVectors = _.map(x, (features) => this[_createSampleVectors](features));

    while (!done) {
      if (iterator === _maxIterator) {
        break;
      }

      done = this[_updateWeights](xVectors, y);
      iterator += 1;
      _debugTraining(`iterator: ${iterator}`);
    }

    if (!done) {
      throw new Error('PLA solution cannot be found');
    }
  }

  get weight() {
    return this[_weight];
  }
}
