import { Vector } from 'vectorious';
import _ from 'lodash';
import Perceptron from './pla';


export default class PocketPLA extends Perceptron {
  constructor(verbose=0) {
    super(verbose);
  }

  _countError(x, y, w, sample) {
    let errorCounts = 0;
    if (!sample || sample <= 0) {
      for (let i=0; i < x.length; i++) {
        if (( x[i].dot(w) * y[i] ) <= 0) {
          errorCounts += 1;
        }
      }
    } else {
      // random sample to speed up pocket
      _.map(_.sampleSize(_.zip(x, y), sample), ([xi, yi]) => {
        if (( xi.dot(w) * yi ) <= 0) {
          errorCounts += 1;
        }
      });
    }
    return errorCounts;
  }

  // override
  _updateWeights(x, y, sample) {
    for (let i=0; i < x.length; i++) {
      if (( x[i].dot(this.w_) * y[i] ) <= 0) {
        if (this.verbose_ > 0) {
          console.log('*************************************');
          console.log(`error case: ${x[i]}, ${this.w_}, ${y[i]}`);
        }
        this.w_ = ( y[i] > 0 ) ? this.w_.add(x[i]) : this.w_.subtract(x[i]);

        // update best weights
        const errorCounts = this._countError(x, y, this.w_, sample);
        if (!this.best_ || (this.best_.errorCounts > errorCounts)) {
          this.best_ = {
            errorCounts,
            w: this.w_,
          };
        }

        if (this.verbose_ > 0) {
          console.log(`new weights: ${this.w_}`);
          console.log('*************************************');
        }
        return true;
      }
    }
    return false;
  }

  fit({ x, y, initWeights, sample, maxIter=100000 } = {}) {
    if (this.verbose_) {
      console.log('Training PocketPLA model ...');
    }
    this._initWeights(initWeights, x[0].length + 1);

    // input pre-process
    const xVectors = _.map(x, (sample) => this._sample2vec(sample));
    const yEncoded = this._encodeLabels(y);
    if (this.verbose_ > 0) {
      console.log(`inputs: ${xVectors}, ${yEncoded}`);
    }

    // update weights
    let resolved = false;
    while (this.iters_ < maxIter) {
      if ((this.iters_ % 1000) === 0) {
        console.log(`iteration: ${this.iters_}/${maxIter}`);
      }
      if (this._updateWeights(xVectors, yEncoded, sample) === false) {
        resolved = true;
        break;
      }
      this.iters_ += 1;
    }

    if (resolved === false) {
      if (this.verbose_ > 0) {
        console.log(`Max iteration -> use best model in pocket`);
      }
      this.w_ = this.best_.w;
    }
  }
}
