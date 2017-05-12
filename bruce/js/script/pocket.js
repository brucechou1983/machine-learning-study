import 'console.table';
import PocketPLA from '../lib/linear_model/pocket';
import trainData from '../../data/w1/not_linear_separable/train_data.10k.json';
import testData from '../../data/w1/not_linear_separable/test_data.10k.json';
import { classificationReport } from '../lib/metrics';

const p = new PocketPLA();

console.log(`******************************************`);
console.time(`training`);
p.fit({ x: trainData.x, y: trainData.y, maxIter: 10000 });
console.timeEnd(`training`);
console.time(`testing`);
const yHat = p.predict(testData.x);
console.timeEnd(`testing`);
console.log('report:');
console.table(classificationReport(testData.y, yHat));
console.log(`******************************************`);


const pSample = new PocketPLA();
// with sample
console.log(`******************************************`);
console.time(`training-sample`);
pSample.fit({ x: trainData.x, y: trainData.y, sample: 100, maxIter: 10000});
console.timeEnd(`training-sample`);
console.time(`testing-sample`);
const yHatSample = pSample.predict(testData.x);
console.timeEnd(`testing-sample`);
console.log('report:');
console.table(classificationReport(testData.y, yHatSample));
console.log(`******************************************`);
