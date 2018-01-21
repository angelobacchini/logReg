#ifndef DATASET_H
#define DATASET_H

#include <iostream>
#include <fstream>
#include <vector>

class dataSet
{
public:

  dataSet(const char* _featuresFile, const char* _labelsFile, int _numExamples, int _numFeatures);
  dataSet(int _numExamples, int _numFeatures);
  ~dataSet();

  int numExamples() const;
  int numFeatures() const;
  
  std::vector<float>* featuresVector() const;
  std::vector<float>* labelsVector() const;

  float* featuresPtr() const;
  float* labelsPtr() const;
  
private:
  int m_numExamples;
  int m_numFeatures;
  std::vector<float>* m_featuresArray;
  std::vector<float>* m_labelsArray;
};

#endif
