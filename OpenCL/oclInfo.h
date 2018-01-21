#ifndef OCLINFO_H
#define OCLINFO_H

#include <iostream>
#include <CL/cl.h>

void oclInfo()
{
  unsigned int i, j; //iterator variables for loops
  cl_platform_id platforms[32]; //an array to hold the IDs of all the platforms, hopefuly there won't be more than 32
  cl_uint num_platforms; //this number will hold the number of platforms on this machine
  char vendor[1024]; //this strirng will hold a platforms vendor
  cl_device_id devices[32]; //this variable holds the number of devices for each platform, hopefully it won't be more than 32 per platform
  cl_uint num_devices; //this number will hold the number of devices on this machine
  char deviceName[1024]; //this string will hold the devices name
  cl_uint numberOfCores; //this variable holds the number of cores of on a device
  cl_long amountOfMemory; //this variable holds the amount of memory on a device
  cl_uint clockFreq; //this variable holds the clock frequency of a device
  cl_ulong maxAlocatableMem; //this variable holds the maximum allocatable memory
  cl_ulong localMem; //this variable holds local memory for a device
  cl_bool  available;//this variable holds if the device is available
  clGetPlatformIDs(32, platforms, &num_platforms); //get the number of platforms    
  std::cout << "Number of platforms: " << num_platforms << std::endl << std::endl;

  for (i = 0; i < num_platforms; i++) // loop through platforms
  {      
    clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, sizeof(vendor), vendor, NULL);      
    clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, sizeof(devices)/sizeof(devices[0]), devices, &num_devices);
    std::cout << "Platform: " << i << std::endl;
    std::cout << "Platform Vendor: " << vendor << std::endl;
    std::cout << "Number of devices: " << num_devices << std::endl;

    for (j = 0; j < num_devices; j++) // loop through devices
    {
      //scan in device information
      clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);
      clGetDeviceInfo(devices[j], CL_DEVICE_VENDOR, sizeof(vendor), vendor, NULL);
      clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(numberOfCores), &numberOfCores, NULL);
      clGetDeviceInfo(devices[j], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(amountOfMemory), &amountOfMemory, NULL);
      clGetDeviceInfo(devices[j], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clockFreq), &clockFreq, NULL);
      clGetDeviceInfo(devices[j], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(maxAlocatableMem), &maxAlocatableMem, NULL);
      clGetDeviceInfo(devices[j], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(localMem), &localMem, NULL);
      clGetDeviceInfo(devices[j], CL_DEVICE_AVAILABLE, sizeof(available), &available, NULL);

      //print out device information
      std::cout << "\tDevice: " << j << std::endl;
      std::cout << "\t\tName:\t\t\t\t " << deviceName << std::endl;
      std::cout << "\t\tVendor:\t\t\t\t " << vendor << std::endl;
      std::cout << "\t\tAvailable:\t\t\t " << available << std::endl;
      std::cout << "\t\tCompute Units:\t\t\t " << numberOfCores << std::endl;
      std::cout << "\t\tClock Frequency:\t\t " << clockFreq << std::endl;
      std::cout << "\t\tGlobal Memory:\t\t\t " << ((double)amountOfMemory / 1048576) << " mb" << std::endl;
      std::cout << "\t\tMax Allocateable Memory:\t " << ((double)maxAlocatableMem / 1048576) << " mb" << std::endl;
      std::cout << "\t\tLocal Memory:\t\t\t " << ((unsigned int)localMem) << " kb" << std::endl << std::endl;
    }
  }
}

#endif