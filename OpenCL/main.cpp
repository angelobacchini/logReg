#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>

#include "global.h"
#include "dataSet.h"
#include "logReg.h"
#include "logRegOcl.h"
#include "oclInfo.h"

int main()
{
  std::chrono::time_point<std::chrono::system_clock> start, stop;
  std::chrono::duration<double> elapsed;

  oclInfo();

  //dataSet* trainingData = new dataSet("./../data/features.txt", "./../data/labels.txt", NUM_EXAMPLES, NUM_FEATURES);
  dataSet* trainingData = new dataSet(NUM_EXAMPLES, NUM_FEATURES);

  logReg* basicLogReg = new logReg(trainingData);
  std::cout << "Running plain C implementation with " << NUM_ITERATIONS << " iterations with learning rate " << LEARNING_RATE << " ..." << std::endl;
  basicLogReg->setWeights();
  start = std::chrono::system_clock::now();
  basicLogReg->run();
  stop = std::chrono::system_clock::now();
  elapsed = stop - start;
  std::vector<float> weights = basicLogReg->getWeights();
  std::cout << "elapsed: " << elapsed.count() << " s" << std::endl;
  std::cout << "weights:";
  for (unsigned int i = 0; i<weights.size(); i++) std::cout << " " << weights.at(i);
  std::cout << std::endl << std::endl;
  delete basicLogReg;

  logRegOcl* oclLogReg = new logRegOcl(trainingData);
  std::cout << "Running openCL implementation with " << NUM_ITERATIONS << " iterations with learning rate " << LEARNING_RATE << " ..." << std::endl;
  oclLogReg->setWeights();
  start = std::chrono::system_clock::now();
  oclLogReg->run();
  stop = std::chrono::system_clock::now();
  elapsed = stop - start;
  std::vector<float> weightsOcl = oclLogReg->getWeights();
  std::cout << "elapsed: " << elapsed.count() << " s" << std::endl;
  std::cout << "weights:";
  for (unsigned int i = 0; i<weightsOcl.size(); i++) std::cout << " " << weightsOcl.at(i);
  std::cout << std::endl << std::endl;
  delete oclLogReg;
  
  delete trainingData;

  return 0;
}
