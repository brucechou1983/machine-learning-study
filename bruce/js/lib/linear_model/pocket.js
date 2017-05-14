import { Vector } from 'vectorious';
import _ from 'lodash';
import Perceptron from './pla';


export default class PocketPLA extends Perceptron {
  constructor(verbose=0) {
    super(verbose);
  }

  _countMaxPLAIter(x, y, sampleRatio = 0.1) {
    // theoretically the maximum iterations for PLA to converge
    // T = ((max ||x||)^2 / (min distance of x to perfect model)^2
    // T can't be calculated since we don't have that perfect model
    // however we can randomly sample some (+1) and (-1) examples
    // and calculate a approximation of T

    const maxSampleIter = 1000;
    let sampleIter = 0;
    let groupedSamples = [];

    // count max length of sampled x
    let maxVectorLength = 0;
    _.map(x, (sampleX) => {
      if (sampleX.magnitude() > maxVectorLength) {
        maxVectorLength = sampleX.magnitude();
      }
    });
    if (this.verbose_ > 0) {
      console.log(`max vector length: ${maxVectorLength}`);
    }

    // group samples
    while (groupedSamples.length !== 2 && sampleIter < maxSampleIter) {
      const samples = _.sampleSize(_.zip(x, y), Math.floor(sampleRatio * x.length));
      groupedSamples = _.partition(samples, (sample) => sample[1] > 0);
      sampleIter += 1;
    }

    // count min distance between grouped samples
    let minGroupedSampleDist = Number.MAX_SAFE_INTEGER;
    _.map(groupedSamples[0], (p) => {
      _.map(groupedSamples[1], (n) => {
        const dist = p[0].subtract(n[0]).magnitude();
        if (dist < minGroupedSampleDist) {
          minGroupedSampleDist = dist;
        }
      })
    });
    if (this.verbose_ > 0) {
      console.log(`min grouped samples distance: ${minGroupedSampleDist}`);
    }

    const maxPLAIter = Math.floor(Math.pow(maxVectorLength, 2) / Math.pow(minGroupedSampleDist / 2, 2));
    console.log(`max PLA iterations: ${maxPLAIter}`);

    return maxPLAIter;
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
    if (this.verbose_ > 0) {
      console.log('Training PocketPLA model ...');
    }
    this._initWeights(initWeights, x[0].length + 1);

    // input pre-process
    const xVectors = _.map(x, (sample) => this._sample2vec(sample));
    const yEncoded = this._encodeLabels(y);
    if (this.verbose_ > 0) {
      console.log(`inputs: ${xVectors}, ${yEncoded}`);
    }

    const maxPLAIter = Math.min(this._countMaxPLAIter(xVectors, yEncoded), maxIter);

    // update weights
    let resolved = false;
    while (this.iters_ < maxIter) {
      if ((this.iters_ % 1000) === 0 && this.iters_ > 0) {
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

