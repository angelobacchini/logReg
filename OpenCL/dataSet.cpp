#include "dataSet.h"
#include <iostream>
#include <fstream>
#include <vector>


// read data from txt files
dataSet::dataSet(const char* _featuresFile, const char* _labelsFile, int _numExamples, int _numFeatures) :
  m_numExamples(_numExamples), m_numFeatures(_numFeatures)
{
  m_featuresArray = new std::vector<float>;
  std::ifstream inputFile(_featuresFile);
  if (inputFile)
  {
    float feature;
    while (inputFile >> feature)
    {
      m_featuresArray->push_back(feature);
    }
  }
  else
  {
    std::cout << "ERROR: Unable to open file " << _featuresFile << std::endl;
  }
  inputFile.close();
  std::cout << m_featuresArray->size() << " features read from " << _featuresFile << std::endl;

  m_labelsArray = new std::vector<float>;
  inputFile.open(_labelsFile);
  if (inputFile)
  {
    float label;
    while (inputFile >> label)
    {
      m_labelsArray->push_back(label);
    }
  }
  else
  {
    std::cout << "ERROR: Unable to open file " << _labelsFile << std::endl;
  }
  inputFile.close();
  std::cout << m_labelsArray->size() << " labels read from " << _labelsFile << std::endl;
  std::cout << std::endl;
}

// generate random data
dataSet::dataSet(int _numExamples, int _numFeatures) :
  m_numExamples(_numExamples), m_numFeatures(_numFeatures)
{
  srand(12345);

  m_featuresArray = new std::vector<float>;
  for (int i = 0; i < m_numExamples*m_numFeatures; i++)
    m_featuresArray->push_back(2*(float)rand()/RAND_MAX - 1);
  std::cout << m_featuresArray->size() << " features generated." << std::endl;


  m_labelsArray = new std::vector<float>;
  for (int i = 0; i < m_numExamples; i++)
    m_labelsArray->push_back(2*(float)rand()/RAND_MAX - 1);
  std::cout << m_labelsArray->size() << " labels generated." << std::endl;
  std::cout << std::endl;
}

dataSet::~dataSet()
{
  delete m_featuresArray;
  delete m_labelsArray;
}

int dataSet::numExamples() const { return m_numExamples; }
int dataSet::numFeatures() const { return m_numFeatures; }

std::vector<float>* dataSet::featuresVector() const { return m_featuresArray; }
std::vector<float>* dataSet::labelsVector() const { return m_labelsArray; }

float* dataSet::featuresPtr() const { return m_featuresArray->data(); }
float* dataSet::labelsPtr() const { return m_labelsArray->data(); }
