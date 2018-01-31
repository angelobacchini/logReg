#ifndef LOGREGOCL_H
#define LOGREGOCL_H

#define CL_HPP_TARGET_OPENCL_VERSION 200
#include <CL/cl2.hpp>
#include "global.h"
#include "dataSet.h"

class logRegOcl
{
public:
  logRegOcl(const dataSet* _dataSet);
  ~logRegOcl();
  logRegOcl(logRegOcl const&) = delete;
  logRegOcl& operator=(logRegOcl const&) = delete;

  void run();
  void setWeights(std::vector<float> _weights);
  void setWeights();
  std::vector<float> getWeights();

private:
  const dataSet* m_dataSet;

  cl_int m_oclStatus; // Used to check the output of each ocl API call

  cl::vector<cl::Platform>* m_platforms;
  cl::vector<cl::Device>* m_devices;
  cl::Device* m_defaultDevice;

  cl::Context* m_context;
  cl::CommandQueue* m_cmdQueue;
  cl::Program* m_program;
  cl::Kernel* m_kernel;

  std::vector<float>* m_weights;
  cl::Buffer* m_featuresBuf;
  cl::Buffer* m_labelsBuf;
  cl::Buffer* m_weightsBuf;
};

#endif
