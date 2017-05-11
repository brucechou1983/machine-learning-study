import Perceptron from '../lib/linear_model/pla';
import trainData from '../../data/train_data.json';
import testData from '../../data/test_data.json';
// import { classificationReport } from './metrics';

const p = new Perceptron();

// train
p.fit(trainData.x, trainData.y);

// in-sample prediction
console.log(`******************************************`);
console.log(`in-sample prediction: ${p.predict(trainData.x)}`);
console.log(`ground truth        : ${trainData.y}`);

// get weights
console.log(`\n******************************************`);
console.log(`final weights: ${p.getParams()}`);

// predict
console.log(`\n******************************************`);
console.log(`prediction  : ${p.predict(testData.x)}`);
console.log(`ground truth: ${testData.y}`);

// console.log(classificationReport(testData.y, yHat));
