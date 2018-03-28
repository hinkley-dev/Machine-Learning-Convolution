#include "layer.h"
#include "layerlinear.h"
#include "matrix.h"
#include "vec.h"

#include <cmath>

LayerLinear::LayerLinear(size_t inputs, size_t outputs, Rand r) :
 Layer(inputs, outputs, r)
{
  m_inputs = inputs;
  m_outputs = outputs;
  weightCount = m_outputs * m_inputs + m_outputs;
}

LayerLinear::~LayerLinear()
{

}

size_t LayerLinear::getInputCount()
{
  return m_inputs;
}

void LayerLinear::activate(const Vec& weights,const Vec& x)
{

  //b will be first "outputs" values in weights

   VecWrapper b(*(Vec*)&weights, 0, m_outputs);


  size_t matrixIndex = m_outputs;




  for(size_t i = 0; i < m_outputs; ++i)
  {
    VecWrapper weightsRow(*(Vec*)&weights, matrixIndex, m_inputs);
    activation[i] = x.dotProduct(weightsRow) + b[i];
    matrixIndex +=weightsRow.size();
  }

}

//the first blame will be y - y_hat


void LayerLinear::backprop(const Vec& weights, Vec& prevBlame)
{



  // size_t matrixIndex = m_outputs -1;
  // for(size_t i = 0; i < m_inputs; ++i)
  // {
  //   double pb = 0.0;
  //   matrixIndex = i;
  //   for(size_t j = 0; j < m_outputs; ++j)
  //   {
  //     pb += weights[matrixIndex] * blame[j];
  //     matrixIndex += m_inputs;
  //   }
  //   prevBlame[i] = pb;
  // }

  //making M
  Matrix M(m_outputs, m_inputs);
  size_t matrixIndex = m_outputs -1;

  for(size_t i = 0; i < m_outputs; ++i)
  {
    for(size_t j = 0; j < m_inputs; ++j)
    {
      matrixIndex++;
      M[i][j] = weights[matrixIndex];
    }
  }

  //calculate prevBlame
  Matrix Blame = vecToMatrix(blame);
  Matrix* M_transposed = M.transpose();
  for(size_t i = 0; i < prevBlame.size(); ++i)
  {
    double val = blame.dotProduct(M_transposed->row(i));
    prevBlame[i] =  val;
  }

  delete M_transposed;


}


void LayerLinear::update_gradient(const Vec& x, Vec& gradient)
{


  //updating b values
  for(size_t i = 0; i < m_outputs; ++i)
  {
    gradient[i] += blame[i];
  }


 size_t matrixIndex = m_outputs;

 //updating matrix values
 for(size_t i = 0; i < m_outputs; ++i)
 {
   for(size_t j = 0; j < m_inputs; ++j)
   {
     gradient[matrixIndex]  += (blame[i] * x[j]);
     matrixIndex++;
   }
 }

 // cout << endl;
 // cout << "In LayerLinear::update_gradient " <<endl;
 // cout << "input vector: ";
 // x.print();
 // cout << endl;
 // cout << "blame vector: ";
 // blame.print();
 // cout << endl;
 // cout << "computed weights gradient: ";
 // gradient.print();
 // cout << endl;

}


void LayerLinear::ordinary_least_squares(const Matrix& X,const Matrix& Y, Vec& weights)
{

    if(X.cols() != m_inputs) throw Ex("Invalid X matrix for input size");
    if(Y.cols() != m_outputs) throw Ex("Invalid Y Matrix for output size");

    Matrix X_originCentered = createOriginCentered(X);
    Matrix Y_originCentered = createOriginCentered(Y);

    //OLS calculation for M
    Matrix* term1 = Matrix::multiply(Y_originCentered, X_originCentered, true, false);

    Matrix* term2 = Matrix::multiply(X_originCentered, X_originCentered, true, false)->pseudoInverse();

    Matrix* M = Matrix::multiply(*term1, *term2, false, false);


    //b calculation...very specific to single features
    Matrix Y_centroids = calculateCentroidVals(Y);
    Matrix X_centroids = calculateCentroidVals(X);

    Matrix* Mx = Matrix::multiply(*M, X_centroids, false, true);

    Vec b = Y_centroids.row(0) - Mx->row(0);

    weights.resize(b.size() + M->row(0).size());
    for(size_t i = 0; i < b.size(); ++i)
    {
      weights[i] = b[i];
    }
    for(size_t i = b.size(), j = 0; i < weights.size(); ++i,++j )
    {
      weights[i] = M->row(0)[j];
    }

    std::cout << "Layer linear OLS weights: " << std::endl;
    weights.print();


    delete term1;
    term1 = NULL;
    delete term2;
    term2 = NULL;
    delete M;
    M = NULL;
    delete Mx;
    Mx = NULL;


}

void LayerLinear::ordinary_least_squares_test()
{


    Matrix X = buildRandom(100,13);
    Vec weights = buildRandom(14);

    //calculate y values with activate
    Matrix Y(100,1);
    for(size_t i = 0; i < Y.rows(); ++i)
    {
      activate(weights, X.row(i));
      for(size_t j = 0; j < activation.size(); ++j)
      {
        Y[i][j] = activation[j];
      }
    }

    addNoise(Y);

    Vec olsWeights;
    ordinary_least_squares(X,Y,olsWeights);

    if(!isNoiseTolerable(weights, olsWeights))
      throw Ex("Claculated and original weights differ too much when noise is added.");

}

void LayerLinear::addNoise(Matrix& M)
{

  double noise_deviation = 0.1;

  for(size_t i = 0; i < M.rows(); ++i)
  {
    for(size_t j = 0; j < activation.size(); ++j)
    {
      M[i][j] += noise_deviation * random.normal();
    }
  }
}

bool LayerLinear::isNoiseTolerable(Vec& computed, Vec& original)
{
  if(computed.size() != original.size()) throw Ex("Cannot check noise tolerance of different sized Vectors");

  double tolerance = 0.05;
  for(size_t i = 0; i < computed.size(); ++i)
  {
    if(abs(computed[i] - original[i]) > tolerance)
      return false;
  }
  return true;
}

Matrix LayerLinear::buildRandom(int rows, int cols)
{
  Matrix m(rows,cols);
  for(size_t r = 0; r < m.rows(); ++r)
  {
    for(size_t c = 0; c < m.cols(); ++c)
    {

      m[r][c] = random.next() % 5;
    }
  }
  return m;
}

Vec LayerLinear::buildRandom(size_t size)
{
  Vec v(size);

  for(size_t i = 0; i < size; ++i)
  {
    v[i] = random.normal();
  }
  return v;
}


Matrix LayerLinear::createOriginCentered(const Matrix& m)
{

  Matrix originCentered(m.rows(), m.cols());

  Matrix centroidVals = calculateCentroidVals(m);

  //mi - avg(m) to fill matrix
  for(size_t i = 0; i < originCentered.rows(); ++i)
  {
    for(size_t j = 0; j < originCentered.cols(); ++j)
    {
      originCentered[i][j] = m[i][j] - centroidVals[0][j];
    }
  }


  return originCentered;
}

Matrix LayerLinear::calculateCentroidVals(const Matrix& m)
{
  Matrix centroidVals(1, m.cols());
  for(size_t i = 0; i < m.cols(); ++i)
  {
    centroidVals[0][i] = m.columnMean(i);
  }
  return centroidVals;
}

Matrix LayerLinear::vecToMatrix(Vec& orig)
{
  Matrix M(1, orig.size());
  for(size_t i = 0; i < orig.size(); ++i)
  {
    M[0][i] = orig[i];
  }
  return M;
}



















//hello
