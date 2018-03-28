// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <exception>
#include <string>
#include <memory>
#include "error.h"
#include "string.h"
#include "rand.h"
#include "matrix.h"
#include "supervised.h"
#include "baseline.h"
#include "layer.h"
#include "layerlinear.h"
#include "neuralnet.h"
#include <algorithm>
#include "imputer.h"
#include "nomcat.h"
#include "normalizer.h"
#include "svg.h"
#include <fstream>
#include <time.h>

using std::cout;
using std::cerr;
using std::string;
using std::auto_ptr;



void testLearner(SupervisedLearner& learner)
{


}


size_t convertToDecimal(const Vec& oneHot)
{
	return oneHot.indexOfMax();
}


void make_features_and_labels(const Matrix& data, Matrix& feats, Matrix& labs)
{
	feats.setSize(data.rows(), data.cols() -1);
	feats.copyBlock(0,0, data, 0,0, data.rows(), data.cols() -1);
	labs.setSize(data.rows(), 1);
	labs.copyBlock(0,0, data, 0, data.cols()-1, data.rows(), 1);
}

void make_training_and_testing(const Matrix& feats, const Matrix& labs, Matrix& trainFeats, Matrix& trainLabs, Matrix& testFeats, Matrix& testLabs, double trainRation)
{
	size_t trainingDataCount = (size_t)(feats.rows() * trainRation);
	size_t testingDataCount = feats.rows() - trainingDataCount;

	trainFeats.setSize(trainingDataCount, feats.cols());
	trainLabs.setSize(trainingDataCount, labs.cols());
	testFeats.setSize(testingDataCount, feats.cols());
	testLabs.setSize(testingDataCount, labs.cols());

	trainFeats.copyBlock(0,0,feats,0,0,trainingDataCount,feats.cols());
	trainLabs.copyBlock(0,0, labs, 0,0, trainingDataCount, labs.cols());

	testFeats.copyBlock(0,0, feats, trainingDataCount, 0, testingDataCount,feats.cols());
	testLabs.copyBlock(0,0,labs, trainingDataCount, 0, testingDataCount, labs.cols());



}

void preprocessData(Matrix& m)
{
	Imputer imputer;
	Normalizer normalizer;
	NomCat nomcat;



	cout << endl;
	//
	//
	 	imputer.train(m);
		Matrix* temp = imputer.transformBatch(m);


		normalizer.train(*temp);
		temp = normalizer.transformBatch(*temp);

		nomcat.train(*temp);
		temp = nomcat.transformBatch(*temp);

		m.copy(*temp);

		delete temp;

}


void testHypothyroid(Rand random)
{
	cout << "start loading data" << endl;
	//load data
	string fn = "data/";
	Matrix hypothyroid_data;
	hypothyroid_data.loadARFF(fn + "hypothyroid.arff");


	Matrix feats;
	Matrix labs;

	//mixup the Data
	for(size_t i = 0; i < hypothyroid_data.rows(); ++i)
	{
		size_t firstRow = random.next(hypothyroid_data.rows());
		size_t secondRow = random.next(hypothyroid_data.rows());

		hypothyroid_data.swapRows(firstRow, secondRow);
	}


	make_features_and_labels(hypothyroid_data, feats, labs);
	preprocessData(feats);
	preprocessData(labs);

	double trainRatio = 0.8;


	Matrix trainFeats;
	Matrix trainLabs;
	Matrix testFeats;
	Matrix testLabs;
	make_training_and_testing(feats, labs, trainFeats,  trainLabs,  testFeats,  testLabs, trainRatio);

	//1. imputer... replaces gaps
	//2. normalizer... normalize the data
	//3. nomcat... replaces nominal vals with vectors of continuous values
	//lecture after momentum.





	cout << "done loading data" << endl;




	if(trainLabs.rows() != trainFeats.rows() || testLabs.rows() != testFeats.rows())
		throw Ex("invalid data in MNIST upload");

	NeuralNet nn(random);
	nn.addLayerLinear(trainFeats.cols(),100);
	nn.addLayerTanh(100);
	nn.addLayerLinear(100,4);
	nn.addLayerTanh(4);

	nn.init_weights();


	NeuralNet nn_mini_batch(random);
	nn_mini_batch.addLayerLinear(trainFeats.cols(),100);
	nn_mini_batch.addLayerTanh(100);
	nn_mini_batch.addLayerLinear(100,4);
	nn_mini_batch.addLayerTanh(4);

	nn_mini_batch.init_weights();



	//size_t testingDataCount = testFeats.rows();



	//GSVG svg(500, 500, 0,0, 200, 1);
	double x_max = 30;
	double y_max = testLabs.rows() / 2;
	GSVG svg(1024, 768,0.0, 0.0, x_max, y_max, 200);

	svg.horizMarks(20);
	svg.vertMarks(20);

	//nn.setTestingData(testFeats, testLabs);

	double learning_rate = 0.03;





	//size_t epochs = 100;
		time_t start;
		time_t current;
		time_t max = 30;
		time_t t0;
		time_t t1;

		time_t prevTime = 0;

		time(&start);
		time(&t0);

		//time(&start);
		size_t miss = 0;
		size_t miss_mini_batch = 0;
		size_t prevMiss = nn.countMisclassifications(testFeats, testLabs);
		size_t prevMiss_mini_batch = nn_mini_batch.countMisclassifications(testFeats, testLabs);
		size_t i = 0;
		while(time(&current) - start < max)
		{
			nn.train(trainFeats, trainLabs, 1, 0.9, learning_rate);
			nn_mini_batch.train(trainFeats, trainLabs, 10, 0.0, learning_rate);
			time(&t1);
			if(t1 - t0 >= 2)
			{
				miss = nn.countMisclassifications(testFeats, testLabs);
				miss_mini_batch = nn_mini_batch.countMisclassifications(testFeats, testLabs);
				svg.line(prevTime, prevMiss, current - start, miss, 1.3, 0x0000ff); //blue
				svg.line(prevTime, prevMiss_mini_batch, current - start, miss_mini_batch, 1.3, 0xff0000); //red
				prevTime = current - start;
				prevMiss = miss;
				prevMiss_mini_batch = miss_mini_batch;
				t0 = t1;
			}
			//convergence detection
			// miss = nn.countMisclassifications(testFeats, testLabs);
			// if((prevMiss - miss)*100/prevMiss < 1.0) break;
			++i;
		}



	//for making first char
	double y_label_pos_x_axis = svg.horizLabelPos();

	double x_label_pos_y_axis = svg.vertLabelPos();
		svg.text(x_max / 2, y_label_pos_x_axis*1.3, "Time (seconds)");
		svg.text(x_label_pos_y_axis*2, y_max / 2, "Missclassifications");
		svg.text(20,testFeats.rows() * 0.4, "Momentum(0.9)");
		svg.line(17, testFeats.rows() * 0.4, 19, testFeats.rows() * 0.4, 1.3, 0x0000ff);

		svg.text(20,testFeats.rows() * 0.3, "Mini batch(10)");
		svg.line(17, testFeats.rows() * 0.3, 19, testFeats.rows() * 0.3, 1.3, 0xff0000);

	std::ofstream s;
	s.exceptions(std::ios::badbit);
	s.open("plot2.svg", std::ios::binary);
	svg.print(s);

}

int main(int argc, char *argv[])
{
	Rand random(123);
	enableFloatingPointExceptions();
	int ret = 1;


	try
	{

		ret = 0;
	}
	catch(const std::exception& e)
	{
		cerr << "An error occurred: " << e.what() << "\n";
	}
	try
	{
		//cout << 325 << endl;
		//convergenceDetection(random);
		testHypothyroid(random);
		// NeuralNet nn(random);
		// nn.addLayerLinear(1,2);
		// nn.addLayerTanh(2);
		// nn.addLayerLinear(2,1);
		// nn.init_weights();
		// Vec in(1);
		// in.fill(0.3);
		// Vec target(1);
		// target.fill(0.7);
		// for(size_t i = 0; i < 3; ++i)
		// {
		// 	nn.train_incremental(in, target );
    //   nn.refine_weights(0.0);
    //    nn.scale_gradient(0.0);
		// }
		//  nn.getWeights().print();





		ret = 0;
	}
	catch(const std::exception& e)
	{
		cerr << "An error occurred: " << e.what() << "\n";
	}
	cout.flush();
	cerr.flush();
	return ret;
}
