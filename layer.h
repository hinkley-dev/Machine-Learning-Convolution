#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include "vec.h"
#include "matrix.h"
#include "rand.h"

class Layer
{

protected:
	Vec activation;
	Vec blame;
	size_t weightCount;
	Rand random;


public:

	Layer(size_t inputs, size_t outputs, Rand r);
	Layer(size_t inputs, Rand r);
	const Vec& getActivation();
	void setBlame(const Vec& _blame);
	size_t getWeightCount();


	virtual ~Layer();

  virtual void ordinary_least_squares(const Matrix& X,const Matrix& Y, Vec& weights) = 0;
	virtual void activate(const Vec& weights,const Vec& x) = 0;
	virtual void backprop(const Vec& weights, Vec& prevBlame) = 0;
	virtual void update_gradient(const Vec& x, Vec& gradient) = 0;

	virtual size_t getInputCount() = 0;

};

#endif
