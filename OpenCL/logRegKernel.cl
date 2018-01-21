#include "global.h"
/* 
__kernel void kernel logReg(__constant float* features, __constant float* labels, __global float* weights)
{
  int i, j, k;  
  float scores[NUM_EXAMPLES];
  float gradients[NUM_FEATURES];
  
  for (i = 0; i < NUM_ITERATIONS; i++)
  {
    // (scores <- trainingSetLabels - sigmoid(trainingSetFeatures%*%weights))
    for(j = 0; j < NUM_EXAMPLES; j++)
    {      
      float score = 0.0f;
      for(k = 0; k < NUM_FEATURES; k++)
        score += features[j*NUM_FEATURES + k] * weights[k];
      
      scores[j] = labels[j] - 1/(1 + exp(-score));
    }
    
    // (gradients <- t(trainingSetFeatures)%*%scores)
    for(k = 0; k < NUM_FEATURES; k++)
    {
      gradients[k] = 0.0f;
      for(j = 0; j < NUM_EXAMPLES; j++)
        gradients[k] += features[j*NUM_FEATURES + k] * scores[j];
    }
      
    // (weights <- weights*(1-LEARNING_RATE*REGULARIZATION) + LEARNING_RATE*gradients)
    for(k = 0; k < NUM_FEATURES; k++)
      weights[k] += gradients[k]*LEARNING_RATE;
  }
}
 */
 
/*  
__kernel void kernel logReg(__constant float* features, __constant float* labels, __global float* weights)
{
  int i, j, k;  
  float score;
  float gradients[NUM_FEATURES];
  
  for (i = 0; i < NUM_ITERATIONS; i++)
  {        
    for(k = 0; k < NUM_FEATURES; k++)
      gradients[k] = 0.0f;
    
    for(j = 0; j < NUM_EXAMPLES; j++)
    {
      score = 0.0f;
      for(k = 0; k < NUM_FEATURES; k++)
        score += features[j*NUM_FEATURES + k] * weights[k];
      
      score = labels[j] - 1/(1 + exp(-score));
      
      for(k = 0; k < NUM_FEATURES; k++)
        gradients[k] += features[j*NUM_FEATURES + k] * score;
    }
      
    for(k = 0; k < NUM_FEATURES; k++)
      weights[k] += gradients[k]*LEARNING_RATE;
  }
}
 */

/* 
__kernel void kernel logReg(__constant float* features, __constant float* labels, __global float* weights)
{
  int localId= get_local_id(0);
  const int localExamplesStart = NUM_EXAMPLES/NUM_WORK_ITEMS*localId;
  
  int i, j, k;  
  float score;
  __local float partialGradients[NUM_FEATURES*NUM_WORK_ITEMS];
  
  for (i = 0; i < NUM_ITERATIONS; i++)
  {        
    for(k = 0; k < NUM_FEATURES; k++)
      partialGradients[localId*NUM_FEATURES + k] = 0.0f;
    
    for(j = 0; j < NUM_EXAMPLES/NUM_WORK_ITEMS; j++)
    {
      score = 0.0f;
      for(k = 0; k < NUM_FEATURES; k++)
        score += features[(localExamplesStart+j)*NUM_FEATURES + k] * weights[k];
      
      score = labels[localExamplesStart+j] - 1/(1 + exp(-score));
      
      for(k = 0; k < NUM_FEATURES; k++)
        partialGradients[localId*NUM_FEATURES + k] += features[(localExamplesStart+j)*NUM_FEATURES + k] * score;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if (localId == 0)
    {
      for(k = 0; k < NUM_FEATURES; k++)
      {
        for (j = 1; j < NUM_WORK_ITEMS; j++)
          partialGradients[k] += partialGradients[j*NUM_FEATURES + k];
      }
      
      for(k = 0; k < NUM_FEATURES; k++)
        weights[k] += partialGradients[k]*LEARNING_RATE;      
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
  }
}

 */

/*  
__kernel void kernel logReg(__constant float* features, __constant float* labels, __global float* weights)
{
  int localId= get_local_id(0);
  const int localExamplesStart = NUM_EXAMPLES/NUM_WORK_ITEMS*localId;
  
  int i, j, k, offset;  
  float score;
  __local float partialGradients[NUM_FEATURES*NUM_WORK_ITEMS];
  
  for (i = 0; i < NUM_ITERATIONS; i++)
  {        
    for(k = 0; k < NUM_FEATURES; k++)
      partialGradients[localId*NUM_FEATURES + k] = 0.0f;
    
    for(j = 0; j < NUM_EXAMPLES/NUM_WORK_ITEMS; j++)
    {
      score = 0.0f;
      for(k = 0; k < NUM_FEATURES; k++)
        score += features[(localExamplesStart+j)*NUM_FEATURES + k] * weights[k];
      
      score = labels[localExamplesStart+j] - 1/(1 + exp(-score));
      
      for(k = 0; k < NUM_FEATURES; k++)
        partialGradients[localId*NUM_FEATURES + k] += features[(localExamplesStart+j)*NUM_FEATURES + k] * score;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
 
    for(offset = NUM_WORK_ITEMS/2; offset > 0; offset >>= 1)
    { 
      for (k = 0; k < NUM_FEATURES; k++)
        partialGradients[localId*NUM_FEATURES + k] += partialGradients[(localId+offset)*NUM_FEATURES + k]; 
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if (localId == 0)
    {
      for(k = 0; k < NUM_FEATURES; k++)
        weights[k] += partialGradients[k]*LEARNING_RATE;      
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
  }
} 
  */

  
__kernel void kernel logReg(__constant float4* features, __constant float* labels, __global float4* weights)
{
  int localId= get_local_id(0);
  const int localExamplesStart = NUM_EXAMPLES/NUM_WORK_ITEMS*localId;
  
  int i, j, k, offset;  
  float score;
  __local float4 partialGradients[NUM_FEATURES/4*NUM_WORK_ITEMS];
  
  for (i = 0; i < NUM_ITERATIONS; i++)
  {        
    for(k = 0; k < NUM_FEATURES/4; k++)
      partialGradients[localId*NUM_FEATURES/4 + k] = 0.0f;
    
    for(j = 0; j < NUM_EXAMPLES/NUM_WORK_ITEMS; j++)
    {
      score = 0.0f;
      for(k = 0; k < NUM_FEATURES/4; k++)
        score += dot(features[(localExamplesStart+j)*NUM_FEATURES/4 + k], weights[k]);
      
      score = labels[localExamplesStart+j] - 1/(1 + exp(-score));
      
      for(k = 0; k < NUM_FEATURES/4; k++)
        partialGradients[localId*NUM_FEATURES/4 + k] += features[(localExamplesStart+j)*NUM_FEATURES/4 + k] * score;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
 
    for(offset = NUM_WORK_ITEMS/2; offset > 0; offset >>= 1)
    { 
      for (k = 0; k < NUM_FEATURES/4; k++)
        partialGradients[localId*NUM_FEATURES/4 + k] += partialGradients[(localId+offset)*NUM_FEATURES/4 + k]; 
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if (localId == 0)
    {
      for(k = 0; k < NUM_FEATURES/4; k++)
        weights[k] += partialGradients[k]*LEARNING_RATE;      
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
  }
}
