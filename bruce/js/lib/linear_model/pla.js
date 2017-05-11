import { Vector } from 'vectorious';
import _ from 'lodash';
import { LabelEncoder } from '../preprocessing';

const w_ = Symbol();
const dim_ = Symbol();
const iters_ = Symbol();
const verbose_ = Symbol();
const le_ = Symbol();

const _updateWeights = Symbol();
const _sample2vec = Symbol();
const _encodeLabels = Symbol();
const _checkClasses = Symbol();

export default class Perceptron {
  constructor(verbose=0) {
    this[iters_] = 0;
    this[verbose_] = verbose;
    this[le_] = new LabelEncoder();
  }

  [_checkClasses]() {
    const classes = this[le_].getParams().classes;
    if (classes.length !== 2) {
      throw new Error(`only binary classification is supported in PLA: ${classes}`);
    }
  }

  [_sample2vec](a) {
    return new Vector(_.concat([1.0], a));
  }

  [_encodeLabels](labels) {
    this[le_].fit(labels);
    this[le_].setParams({ codes: [-1, 1]});
    this[_checkClasses]();
    return this[le_].transform(labels);
  }

  [_updateWeights](x, y) {
    for (let i=0; i < x.length; i++) {
      if (( x[i].dot(this[w_]) * y[i] ) <= 0) {
        if (this[verbose_] > 0) {
          console.log('*************************************');
          console.log(`error case: ${x[i]}, ${this[w_]}, ${y[i]}`);
        }
        this[w_] = ( y[i] > 0 ) ? this[w_].add(x[i]) : this[w_].subtract(x[i]);
        if (this[verbose_] > 0) {
          console.log(`new weights: ${this[w_]}`);
          console.log('*************************************');
        }
        return true;
      }
    }
    return false;
  }

  fit(x, y, initWeights, maxIter=10000) {
    if (this[verbose_]) {
      console.log('Training PLA model ...');
    }
    this[dim_] = x[0].length + 1;

    // weights initialization
    if(!initWeights){
      this[w_] = Vector.zeros(this[dim_]);
    } else {
      if (initWeights.length !== this[dim_]) {
        throw new Error(`initial weights: ${initWeights}`);
      }
      this[w_] = new Vector(initWeights);
    }
    
    if (this[verbose_] > 0) {
      console.log(`initialize weights: ${JSON.stringify(this[w_])}`);
    }

    // input pre-process
    const xVectors = _.map(x, (sample) => this[_sample2vec](sample));
    const yEncoded = this[_encodeLabels](y);
    if (this[verbose_] > 0) {
      console.log(`inputs: ${xVectors}, ${yEncoded}`);
    }

    // update weights
    let resolved = false;
    while (this[iters_] < maxIter) {
      if (this[_updateWeights](xVectors, yEncoded) === false) {
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
      xVector) => (xVector.dot(this[w_]) < 0) ? this[le_].inverseTransform(0) : this[
        le_].inverseTransform(1));
  }

  getParams() {
    return this[w_];
  }
}

