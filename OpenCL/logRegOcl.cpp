#include <iostream>
#include "logRegOcl.h"

logRegOcl::logRegOcl(const dataSet* _dataSet) :
  m_dataSet(_dataSet)
{
  m_weights = new std::vector<float>(m_dataSet->numFeatures()); // vector that will store weights (read and written by openCl kernel)

  // setup platfrom and device
  m_platforms = new cl::vector<cl::Platform>;
  cl::Platform::get(m_platforms);
  if (m_platforms->size() == 0) {
    std::cout << " No platforms found. Check OpenCL installation!\n";
    exit(1);
  }
  cl::Platform default_platform = m_platforms->at(OCL_PLATFORM);
  std::cout << "Using platform: " << default_platform.getInfo<CL_PLATFORM_NAME>() << "\n";

  m_devices = new cl::vector<cl::Device>;
  default_platform.getDevices(CL_DEVICE_TYPE_ALL, m_devices);
  if (m_devices->size() == 0) {
    std::cout << " No devices found. Check OpenCL installation!\n";
    exit(1);
  }
  m_defaultDevice = new cl::Device(m_devices->at(OCL_DEVICE));
  std::cout << "Using device: " << m_defaultDevice->getInfo<CL_DEVICE_NAME>() << "\n";

  m_context = new cl::Context({ *m_defaultDevice });

  // source OpenCL kernel
  cl::Program::Sources sources;
  std::ifstream kernelFile("./logRegKernel.cl");
  if (!kernelFile) { std::cout << "error when reading logRegKernel.cl" << std::endl; }
  std::string kernelCode(std::istreambuf_iterator<char>(kernelFile), (std::istreambuf_iterator<char>()));
  sources.push_back({ kernelCode.c_str(),kernelCode.length() });

  // build kernel
  m_program = new cl::Program(*m_context, sources);
  if (m_program->build({*m_defaultDevice }, "-I ./../src") != CL_SUCCESS)
  {
    std::cout << " Error building: " << m_program->getBuildInfo<CL_PROGRAM_BUILD_LOG>(*m_defaultDevice) << "\n";
    exit(1);
  }

  // allocate buffers
  m_featuresBuf = new cl::Buffer(*m_context, CL_MEM_READ_ONLY, NUM_EXAMPLES*NUM_FEATURES*sizeof(float));
  m_labelsBuf = new cl::Buffer(*m_context, CL_MEM_READ_ONLY, NUM_EXAMPLES*sizeof(float));
  m_weightsBuf = new cl::Buffer(*m_context, CL_MEM_READ_WRITE, NUM_FEATURES *sizeof(float));

  // create OpenCl queue
  m_cmdQueue = new cl::CommandQueue(*m_context, *m_defaultDevice);

  // setup kernel
  m_kernel = new cl::Kernel(*m_program, "logReg");  
  m_kernel->setArg(0, *m_featuresBuf);
  m_kernel->setArg(1, *m_labelsBuf);
  m_kernel->setArg(2, *m_weightsBuf);
}

logRegOcl::~logRegOcl()
{
  delete m_kernel;
  delete m_program;
  delete m_featuresBuf;
  delete m_labelsBuf;
  delete m_weightsBuf;
  delete m_weights;
  delete m_cmdQueue;
  delete m_context;
  delete m_devices;
  delete m_platforms;
}

void logRegOcl::run()
{
  // populate buffers and enqueue kernels
  m_cmdQueue->enqueueWriteBuffer(*m_featuresBuf, CL_FALSE, 0, NUM_EXAMPLES*NUM_FEATURES*sizeof(float), m_dataSet->featuresPtr());
  m_cmdQueue->enqueueWriteBuffer(*m_labelsBuf, CL_FALSE, 0, NUM_EXAMPLES*sizeof(float), m_dataSet->labelsPtr());
  m_cmdQueue->enqueueWriteBuffer(*m_weightsBuf, CL_FALSE, 0, NUM_FEATURES*sizeof(int), m_weights->data());
  m_cmdQueue->enqueueNDRangeKernel(*m_kernel, cl::NullRange, cl::NDRange(NUM_WORK_ITEMS), cl::NDRange(NUM_WORK_ITEMS));
  m_cmdQueue->enqueueReadBuffer(*m_weightsBuf, CL_TRUE, 0, NUM_FEATURES*sizeof(float), m_weights->data());
}

void logRegOcl::setWeights(std::vector<float> _weights)
{
  for (unsigned int i = 0; i < m_weights->size(); i++) { (*m_weights)[i] = _weights.at(i); }
}

void logRegOcl::setWeights()
{
  for (unsigned int i = 0; i < m_weights->size(); i++) { (*m_weights)[i] = 0.0f; }
}

std::vector<float> logRegOcl::getWeights() { return *m_weights; }
