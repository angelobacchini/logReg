all: intel amd nvidia

is_64=$(shell s=`uname -m`; if (echo $$s | grep x86_64 > /dev/null); then echo 1; fi)

CFLAGS=-Wall -W -O2

# AMD
PATH_TO_AMD_INC=/opt/AMDAPP/include/
ifeq ($(is_64), 1)
PATH_TO_AMD_LIB=/opt/AMDAPP/lib/x86_64/
else
PATH_TO_AMD_LIB=/opt/AMDAPP/lib/x86/
endif

# Intel
PATH_TO_INTEL_INC=/usr/include/
ifeq ($(is_64), 1)
PATH_TO_INTEL_LIB=/usr/lib64/
endif

# NVIDIA
PATH_TO_NVIDIA_INC=/usr/local/cuda/include/
PATH_TO_NVIDIA_LIB=/usr/lib/

EXEC = run
SOURCES = $(wildcard *.cpp)
INTEL_OBJS = $(SOURCES:.cpp=.intel)
AMD_OBJS = $(SOURCES:.cpp=.amd)
NVIDIA_OBJS = .$(SOURCES:.cpp=.nvidia)

amd:$(AMD_OBJS)
	g++ $(CFLAGS) -L$(PATH_TO_AMD_LIB) -Wl,-rpath,$(PATH_TO_AMD_LIB) $(AMD_OBJS) -o $(join $(EXEC),$@) -lOpenCL
intel:$(INTEL_OBJS)
	g++ $(CFLAGS) -L$(PATH_TO_INTEL_LIB) -Wl,-rpath,$(PATH_TO_INTEL_LIB) $(INTEL_OBJS) -o $(join $(EXEC),$@) -lOpenCL
nvidia:$(NVIDIA_OBJS)
	g++ $(CFLAGS) -L$(PATH_TO_NVIDIA_LIB) $(NVIDIA_OBJS) -o $(join $(EXEC),$@) -lOpenCL

%.amd:%.cpp
	g++ -c -I$(PATH_TO_AMD_INC) $(CFLAGS) -MMD $< -o $@
%.intel:%.cpp
	g++ -c -I$(PATH_TO_INTEL_INC) $(CFLAGS) -MMD $< -o $@
%.nvidia:%.cpp
	g++ -c -I$(PATH_TO_NVIDIA_INC) $(CFLAGS) -MMD $< -o $@

clean:
	rm -f run* *.amd *.intel *.nvidia *.d

-include *.d
