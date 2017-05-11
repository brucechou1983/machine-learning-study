import { Vector } from 'vectorious';
import _ from 'lodash';

const _w = Symbol();
const _dim = Symbol();
const _labelType= Symbol();
const _labelMap = Symbol();
const _iters = Symbol();
const _verbose = Symbol();
const _updateWeights = Symbol();
const _sample2vec = Symbol();
const _labels2vec = Symbol();

class Perceptron {
  constructor(verbose=0) {
    this[_iters] = 0;
    this[_verbose] = verbose;
  }

  [_sample2vec](a) {
    return new Vector(_.concat([1.0], a));
  }

  [_labels2vec](labels) {
    this[_labelType] = _.uniq(labels);
    if (this[_labelType].length !== 2) {
      throw new Error(`only binary classification is supported in PLA: ${this[_labelType]}`);
    }

    this[_labelMap] = {};
    this[_labelMap][this[_labelType][0]] = -1.0;
    this[_labelMap][this[_labelType][1]] = 1.0;

    return _.map(labels, (l) => this[_labelMap][l]);
  }

  [_updateWeights](x, y) {
    for (let i=0; i < x.length; i++) {
      if (( x[i].dot(this[_w]) * y[i] ) <= 0) {
        if (this[_verbose] > 0) {
          console.log('*************************************');
          console.log(`error case: ${x[i]}, ${this[_w]}, ${y[i]}`);
        }
        this[_w] = ( y[i] > 0 ) ? this[_w].add(x[i]) : this[_w].subtract(x[i]);
        if (this[_verbose] > 0) {
          console.log(`new weights: ${this[_w]}`);
          console.log('*************************************');
        }
        return true;
      }
    }
    return false;
  }

  train(x, y, initWeights, maxIter=10000) {
    console.log('Training PLA model ...');
    this[_dim] = x[0].length + 1;

    // weights initialization
    if(!initWeights){
      this[_w] = Vector.zeros(this[_dim]);
    } else {
      if (initWeights.length !== this[_dim]) {
        throw new Error(`initial weights: ${initWeights}`);
      }
      this[_w] = new Vector(initWeights);
    }
    
    if (this[_verbose] > 0) {
      console.log(`initialize weights: ${JSON.stringify(this[_w])}`);
    }

    // input pre-process
    const xVectors = _.map(x, (sample) => this[_sample2vec](sample));
    const yVector = this[_labels2vec](y);
    if (this[_verbose] > 0) {
      console.log(`inputs: ${xVectors}, ${yVector}`);
    }

    // update weights
    let resolved = false;
    while (this[_iters] < maxIter) {
      if (this[_updateWeights](xVectors, yVector) === false) {
        resolved = true;
        break;
      }
    }

    if (resolved === false) {
      throw new Error('PLA solution cannot be found');
    }
  }

  predict(x) {
    const xVectors = _.map(x, (sample) => this[_sample2vec](sample));
    return _.map(xVectors, (
      xVector) => (xVector.dot(this[_w]) < 0) ? this[_labelType][0] : this[_labelType][1]);
  }

  get weights() {
    return this[_w];
  }
}

export default Perceptron;

