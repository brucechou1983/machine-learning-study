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
      if (this._updateWeights(xVectors, yEncoded) === false) {
        resolved = true;
        break;
      }

      // update best weights
      if (this.iters_ >= maxPLAIter) {
        const errorCounts = this._countError(xVectors, yEncoded, this.w_, sample);
        if (!this.best_ || (this.best_.errorCounts > errorCounts)) {
          this.best_ = {
            errorCounts,
            w: this.w_,
          };
        }
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

