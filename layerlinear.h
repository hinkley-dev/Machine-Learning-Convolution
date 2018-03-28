#ifndef LAYERLINEAR_H
#define LAYERLINEAR_H

#include <vector>
#include "vec.h"
#include "matrix.h"

using namespace std;

class LayerLinear : public Layer
{
  size_t m_inputs;
  size_t m_outputs;



private:
  Matrix createOriginCentered(const Matrix& m);
  Matrix calculateCentroidVals(const Matrix& m);

  bool isNoiseTolerable(Vec& computed, Vec& original);
  void addNoise(Matrix& M);
  Matrix vecToMatrix(Vec& orig);



public:
	LayerLinear(size_t inputs, size_t outputs, Rand r);

	~LayerLinear();
   Vec buildRandom(size_t size);
   Matrix buildRandom(int rows, int cols);


	virtual void activate(const Vec& weights,const Vec& x);
  virtual void backprop(const Vec& weights, Vec& prevBlame);
	virtual void update_gradient(const Vec& x, Vec& gradient);
  virtual size_t getInputCount();

  void ordinary_least_squares(const Matrix& X,const Matrix& Y, Vec& weights);
  void ordinary_least_squares_test();
};



#endif
