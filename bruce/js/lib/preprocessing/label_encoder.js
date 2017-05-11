import _ from 'lodash';

const classes_ = Symbol();
const codes_ = Symbol();
const _checkCodes = Symbol();

export default class LabelEncoder {

  fitTransform(y) {
    this.fit(y);
    return this.transform(y);
  }

  fit(y) {
    this[codes_] = Array.apply(null, {length: y.length}).map(Number.call, Number);
    this[classes_] = _.uniq(y);
  }

  [_checkCodes]() {
    if (this[codes_].length !== this[classes_].length) {
      throw new Error(`Unmatched classes and codes: \n${this[classes_]}\n${this[codes_]}`);
    }
  }

  transform(y) {
    this[_checkCodes]();
    const handle = (label) => this[codes_][_.indexOf(this[classes_], label)];

    if (Array.isArray(y)) {
      return _.map(y, handle);
    }
    return handle(y);
  }

  inverseTransform(encoded_y) {
    this[_checkCodes]();
    const handle = (code) => this[classes_][code];

    if (Array.isArray(encoded_y)) {
      return _.map(encoded_y, handle);
    }
    return handle(encoded_y);
  }

  getParams() {
    return {
      classes: this[classes_],
      codes: this[codes_],
    }
  }

  setParams(params) {
    if ('codes' in params) {
      this[codes_] = params.codes;
    }
  }
}

