#include <math.h>
#include "logReg.h"

logReg::logReg(const dataSet* _dataSet) :
  m_dataSet(_dataSet)
{
  m_weights = new std::vector<float>;
  m_weights->resize(NUM_FEATURES, 0);
}

logReg::~logReg()
{
  delete m_weights;
}

void logReg::run()
{
  float* features = m_dataSet->featuresVector()->data();
  float* labels = m_dataSet->labelsVector()->data();
  float* weights = m_weights->data();

  int i, j, k;  
  float scores[NUM_EXAMPLES];
  float gradients[NUM_FEATURES];

  for (i = 0; i < NUM_ITERATIONS; i++)
  {
    // (scores <- trainingSetLabels - sigmoid(trainingSetFeatures%*%weights))
    for (j = 0; j < NUM_EXAMPLES; j++)
    {
      float score = 0.0f;
      for (k = 0; k < NUM_FEATURES; k++)
        score += features[j*NUM_FEATURES + k] * weights[k];

      scores[j] = labels[j] - 1 / (1 + exp(-score));
    }

    // (gradients <- t(trainingSetFeatures)%*%scores)
    for (k = 0; k < NUM_FEATURES; k++)
    {
      gradients[k] = 0.0f;
      for (j = 0; j < NUM_EXAMPLES; j++)
        gradients[k] += features[j*NUM_FEATURES + k] * scores[j];
    }

    // (weights <- weights*(1-LEARNING_RATE*REGULARIZATION) + LEARNING_RATE*gradients)
    for (k = 0; k < NUM_FEATURES; k++)
      weights[k] += gradients[k] * LEARNING_RATE;
  }
}

void logReg::setWeights(std::vector<float> _weights)
{
  for (unsigned int i = 0; i < m_weights->size(); i++) { (*m_weights)[i] = _weights.at(i); }
}

void logReg::setWeights()
{
  for (unsigned int i = 0; i < m_weights->size(); i++) { (*m_weights)[i] = 0.0f; }
}

std::vector<float> logReg::getWeights() { return *m_weights; }
