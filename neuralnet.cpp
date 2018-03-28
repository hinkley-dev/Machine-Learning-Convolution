#include "neuralnet.h"
#include "supervised.h"

NeuralNet::NeuralNet(Rand r) : random(r)
{

}

NeuralNet::~NeuralNet()
{
  for(size_t i = 0; i < layers.size(); i++)
    		delete(layers[i]);
}

const char* NeuralNet::name()
{
  return "neural network";
}

Vec NeuralNet::getWeights()
{
  return weights;
}

void NeuralNet::train(Matrix& trainFeats, Matrix& trainLabs)
{

  //stochastic_train(trainFeats, trainLabs, 0.8);
}


void NeuralNet::train(Matrix& trainFeats, Matrix& trainLabs, size_t batch_size, double momentum, double learning_rate)
{

  size_t trainingDataCount = trainFeats.rows();

  size_t *randomIndicies= new size_t[trainingDataCount];
  for(size_t j = 0; j < trainingDataCount; ++j)
    randomIndicies[j] = j;


   random_shuffle(&randomIndicies[0],&randomIndicies[trainingDataCount]);


  for(size_t j = 0; j < trainingDataCount; ++j)
  {
    double lr = learning_rate;
    size_t row = randomIndicies[j];


     train_incremental(trainFeats.row(row), trainLabs.row(row) );

     if(j % batch_size == 0 && j > 0)
     {
       refine_weights(lr);
       scale_gradient(momentum);
     }

   }

  delete[] randomIndicies;
}

size_t NeuralNet::countMisclassifications(const Matrix& features, const Matrix& labels)
{
	if(features.rows() != labels.rows())
		throw Ex("Mismatching number of rows");
	size_t mis = 0;
	for(size_t i = 0; i < features.rows(); i++)
	{
		const Vec& pred = predict(features[i]);
		const Vec& lab = labels[i];
		size_t predVal = pred.indexOfMax();
    size_t labVal = lab.indexOfMax();
			if(predVal != labVal)
			{
				mis++;
			}

	}
	return mis;
}


float NeuralNet::root_mean_squared_error(Matrix& features, Matrix& labels)
{
  float rmse = 0.0;
  float sse = sum_squared_error(features, labels);
  rmse = sqrt(sse / (features.rows()));
  return rmse;
}

void NeuralNet::setTestingData(const Matrix& testFeats, const Matrix& testLabs)
{
  testingFeatures.copy(testFeats);
  testingLabels.copy(testLabs);
}








void NeuralNet::scale_gradient(double scale)
{
  gradient *= scale;
}

void NeuralNet::train_incremental(const Vec& feat, const Vec& lab)
{

  predict(feat);
  backprop(lab);
  update_gradient(feat);

}



const Vec& NeuralNet::predict(const Vec& in)
{

  Vec layerInputs(in);
  size_t startWeight = 0;
  for(size_t i = 0; i < layers.size(); ++i)
  {
    VecWrapper layerWeights(weights, startWeight, layers[i]->getWeightCount());
    layers[i]->activate(layerWeights, layerInputs);

    layerInputs.copy(layers[i]->getActivation());
    startWeight += layers[i]->getWeightCount();
  }

  return layers[layers.size() -1]->getActivation();
}

void NeuralNet::backprop(const Vec& targetVals)
{
  Vec finalActivation(layers[layers.size() -1]->getActivation());
  Vec initialBlame(finalActivation.size());


  for(size_t i = 0; i < initialBlame.size(); ++i)
  {
    initialBlame[i] = targetVals[i] - finalActivation[i];
  }

  layers[layers.size() -1]->setBlame(initialBlame);
  size_t startWeight = weights.size();
  Vec prevBlame(initialBlame);

  for(size_t i = layers.size() - 1; i > 0; --i)
  {

    //build the weights
    startWeight -= layers[i]->getWeightCount();
    VecWrapper layerWeights(weights, startWeight,layers[i]->getWeightCount());

    layers[i]->setBlame(prevBlame);
    prevBlame.resize(layers[i]->getInputCount());
    prevBlame.fill(0.0);
    layers[i]->backprop(layerWeights, prevBlame);
  }
  layers[0]->setBlame(prevBlame);

}

void NeuralNet::update_gradient(const Vec& x)
{

  if(&x == nullptr) throw Ex("input to update gradient is null");
  if(x.size() == 0) throw Ex("input is not the right size");
  Vec in(x);
  size_t startGradient = 0;

  for(size_t i = 0; i < layers.size(); ++i)
  {

    VecWrapper layerGradient(gradient, startGradient, layers[i]->getWeightCount());
    layers[i]->update_gradient(in, layerGradient);

    //copying over the gradient
    for(size_t j = startGradient, k = 0; k < layerGradient.size(); ++j, ++k)
    {
      gradient[j] = layerGradient[k];
    }

    in.copy(layers[i]->getActivation());
    startGradient += layerGradient.size();
  }

}

void NeuralNet::init_weights()
{
  size_t startWeight = 0;

  for(size_t layerIndex = 0 ; layerIndex < layers.size(); ++layerIndex)
  {
   for(size_t i = startWeight; i < startWeight + layers[layerIndex]->getWeightCount(); ++i)
    {
      if(i >= weights.size()) throw Ex("Math error in init weights");
      if( layers[layerIndex]->getInputCount() <= 0) throw Ex("Math error in init weights (trying to divide by 0)");
      weights[i] = max(0.03, 1.0 / layers[layerIndex]->getInputCount()) * random.normal();
    }
    startWeight += layers[layerIndex]->getWeightCount();

  }
  if(startWeight != weights.size()) throw Ex("Error, not all weights initialized");
  //weights.copy({0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3});
  gradient.fill(0.0);

}

void NeuralNet::refine_weights(double learning_rate)
{
  weights.addScaled(learning_rate, gradient);

}

void NeuralNet::addLayerLinear(size_t in, size_t out)
{
  layers.push_back(new LayerLinear(in, out, random));
  weights.resize(weights.size() + layers[layers.size()-1]->getWeightCount());
  gradient.resize(weights.size());
}

void NeuralNet::addLayerTanh(size_t in)
{
  layers.push_back(new LayerTanh(in, random));

}

size_t NeuralNet::effective_batch_size(double momentum)
{
  return (size_t)(1/(1-momentum));
}
