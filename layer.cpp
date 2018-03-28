#include "layer.h"

Layer::Layer(size_t inputs, size_t outputs, Rand r) :
  activation(outputs), blame(outputs), random(r)
{
  activation.fill(0.0);
  blame.fill(0.0);
}

Layer::Layer(size_t inputs, Rand r) :
  activation(inputs), blame(inputs), random(r)
{
  activation.fill(0.0);
  blame.fill(0.0);
}

Layer::~Layer()
{

}

const Vec& Layer::getActivation()
{
  return activation;
}

void Layer::setBlame(const Vec& _blame)
{
  blame.copy(_blame);
}

size_t Layer::getWeightCount()
{
  return weightCount;
}
