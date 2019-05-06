# Copyright (C) 2013-2016 Altera Corporation, San Jose, California, USA. All rights reserved.
# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to
# whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.
# 
# This agreement shall be governed in all respects by the laws of the State of California and
# by the laws of the United States of America.
# This is a GNU Makefile.

# You must configure ALTERAOCLSDKROOT to point the root directory of the Intel(R) FPGA SDK for OpenCL(TM)
# software installation.
# See http://www.altera.com/literature/hb/opencl-sdk/aocl_getting_started.pdf 
# for more information on installing and configuring the Intel(R) FPGA SDK for OpenCL(TM).
VERBOSE=1
ifeq ($(VERBOSE),1)
ECHO := 
else
ECHO := @
endif

# Where is the Intel(R) FPGA SDK for OpenCL(TM) software?
# ifeq ($(wildcard $(ALTERAOCLSDKROOT)),)
# $(error Set ALTERAOCLSDKROOT to the root directory of the Intel(R) FPGA SDK for OpenCL(TM) software installation)
# endif
# ifeq ($(wildcard $(ALTERAOCLSDKROOT)/host/include/CL/opencl.h),)
# $(error Set ALTERAOCLSDKROOT to the root directory of the Intel(R) FPGA SDK for OpenCL(TM) software installation.)
# endif

TIMER_PATH=../timer
AOC := aoc

# OpenCL compile and link flags.
AOCL_COMPILE_CONFIG := $(shell aocl compile-config ) -I$(ALTERAOCLSDKROOT)/board/nalla_pcie/software/include
AOCL_LINK_CONFIG := $(shell aocl link-config ) -lhook -lnalla_pcie_mmd
INTELDIR := /opt/intel/opencl-1.2-sdk-6.2.0.1760

CXXFLAGS += -I$(ALTERAOCLSDKROOT)/board/nalla_pcie/software/include -lhook -lnalla_pcie_mmd

# Compilation flags
ifeq ($(DEBUG),1)
CXXFLAGS += -g
else
CXXFLAGS += -O3
endif

# Compiler
CXX ?= g++ 

ifeq ($(CXX),icpc)
CXXFLAGS += -openmp -xavx
else
CXXFLAGS += -fopenmp
endif

# Make it all!
all : $(TARGET_DIR)/$(TARGET) $(TARGET_DIR)/$(CPU_TARGET)

# Host executable target.
$(TARGET_DIR)/$(TARGET) : Makefile $(SRCS) $(INCS) $(TARGET_DIR)
	$(ECHO)$(CXX) $(CPPFLAGS) $(CXXFLAGS) -fPIC $(foreach D,$(INC_DIRS),-I$D) \
			$(AOCL_COMPILE_CONFIG) $(SRCS) $(AOCL_LINK_CONFIG) \
			$(foreach D,$(LIB_DIRS),-L$D) \
			$(foreach L,$(LIBS),-l$L) \
			-o $(TARGET_DIR)/$(TARGET)

$(TARGET_DIR) :
	$(ECHO)mkdir $(TARGET_DIR)

$(TARGET_DIR)/%.aocx :
	aoc --board p385a_sch_ax115 -o $@ $(AOCL_EMULATE) $(AOCL_APP_FLAG) $(AOCL_APP_CUSTOM_FLAG) device/$(*F).cl

emulate : CFLAGS += -DEMULATOR 

emulate : $(TARGET_DIR)/$(DEBUG_AOCX_OUT) $(TARGET_DIR)/$(TARGET)
	rm -f run.log
	env CL_CONTEXT_EMULATOR_DEVICE_ALTERA=1 $(TARGET_DIR)/$(TARGET) 

$(TARGET_DIR)/$(DEBUG_AOCX_OUT) : Makefile $(SRCS) $(INCS) $(TARGET_DIR)
	@echo "-I- : Emulating $(KERNEL)"
	$(AOC) -v -march=emulator $(KERNEL) -o $(TARGET_DIR)/$(DEBUG_AOCX_OUT) -DEMULATOR \
	-I $(TIMER_PATH)/src/c -l timer.aoclib -L $(TIMER_PATH)

emu-test :
	env CL_CONTEXT_EMULATOR_DEVICE_ALTERA=1 ./bin/host
	
# Standard make targets
clean :
	$(ECHO)rm -f $(TARGET_DIR)/$(TARGET)

.PHONY : all clean
