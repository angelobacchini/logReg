#ifndef LOGREG_H
#define LOGREG_H

#include "global.h"
#include "dataSet.h"

class logReg
{
public:
  logReg(const dataSet* _dataSet);
  ~logReg();

  void run();
  void setWeights(std::vector<float> _weights);
  void setWeights();
  std::vector<float> getWeights();

private:
  const dataSet* m_dataSet;
  std::vector<float>* m_weights;
};

#endif