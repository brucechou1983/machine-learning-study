import 'console.table';
import Perceptron from '../lib/linear_model/pla';
import trainData from '../../data/w1/linear_separable/train_data.json';
import testData from '../../data/w1/linear_separable/test_data.json';
import { classificationReport } from '../lib/metrics';

const p = new Perceptron();

// train
p.fit({ x: trainData.x, y: trainData.y });

// get weights
console.log(`******************************************`);
console.log(`final weights: ${p.getParams()}`);

// predict
const yHat = p.predict(testData.x);

console.log(`******************************************`);
console.table(classificationReport(testData.y, yHat));
