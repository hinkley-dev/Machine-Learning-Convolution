#ifndef LAYERTANH_H
#define LAYERTANH_H

#include "vec.h"
#include "matrix.h"
#include "rand.h"
#include "layer.h"
#include <cmath>

using namespace std;

class LayerTanh : public Layer
{

protected:
  size_t m_inputs;



public:

	LayerTanh(size_t inputs, Rand r);
	~LayerTanh();

  virtual void ordinary_least_squares(const Matrix& X,const Matrix& Y, Vec& weights);
	virtual void activate(const Vec& weights,const Vec& x);
	virtual void backprop(const Vec& weights, Vec& prevBlame);
	virtual void update_gradient(const Vec& x, Vec& gradient);

	virtual size_t getInputCount();

};

#endif
