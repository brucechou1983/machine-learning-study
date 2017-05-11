import Perceptron from '../lib/linear_model/pla';
import trainData from '../../data/train_data.json';
import testData from '../../data/test_data.json';

const p = new Perceptron();

// train
p.train(trainData.x, trainData.y);

// in-sample prediction
console.log(`******************************************`);
console.log(`in-sample prediction: ${p.predict(trainData.x)}`);
console.log(`ground truth        : ${trainData.y}`);

// predict
console.log(`******************************************`);
console.log(`prediction  : ${p.predict(testData.x)}`);
console.log(`ground truth: ${testData.y}`);

// TODO
// import { classificationReport } from './metrics';
// console.log(classificationReport(testData.y, yHat));
