############################################################################################################
# Author      : Anshuman Verma (anshuman@vt.edu) 
# Date 	      : June 2nd, 2016 
# Description : Took the makefile created by Jehandad (probably Altera Makefile) and updated for targets for 
#				emulation, compile, and creating database in scratch directory
############################################################################################################

############################################################################################################
#User Options
############################################################################################################
DEBUG := 1
VERBOSE := 1
O := temp
K := src/matrix_mul.cl
B := ${BOARD}
M := 64
N := 64
P := 1
ifeq ($(VERBOSE),1)
ECHO := 
else
ECHO := @
endif

TIMER_PATH=../timer

############################################################################################################
# Where is the Altera SDK for OpenCL software? Find the SDK, AOCL before compilation
############################################################################################################
ifeq ($(wildcard $(ALTERAOCLSDKROOT)),)
$(error Set ALTERAOCLSDKROOT to the root directory of the Altera SDK for OpenCL software installation)
endif
ifeq ($(wildcard $(ALTERAOCLSDKROOT)/host/include/CL/opencl.h),)
$(error Set ALTERAOCLSDKROOT to the root directory of the Altera SDK for OpenCL software installation.)
endif
AOCL := aocl
AOC  := aoc
# OpenCL compile and link flags.
AOCL_COMPILE_CONFIG := $(shell $(AOCL) --noverbose compile-config )
AOCL_LINK_CONFIG := $(shell $(AOCL) --noverbose link-config )
ifeq ($(AOCL_COMPILE_CONFIG),)
$(error AOCL is not found, Check variable PATH)
endif

############################################################################################################
# GCC, CPP compiler and options
# Compilation flags
############################################################################################################
ifeq ($(DEBUG),1)
CXXFLAGS += -g -std=c++0x
else
CXXFLAGS += -O2
endif
# Compiler
CXX := g++


############################################################################################################
# Target, Include files, and CPP/C files
############################################################################################################
TARGET := mm
TARGET_DIR := bin

# Directories
INC_DIRS := 
LIB_DIRS :=  
#LIB_DIRS :=  /opt/aalsdk/aalsdk-5.0.2/lib

# Files
INCS := $(wildcard src/*.h)
SRCS := $(wildcard src/*.cpp )
LIBS := 


KERNEL := $(K) 
ifeq ($(O),)
O := $(K)
endif 
AOCX_OUT := $(O).aocx
AOCO_OUT := $(O).aoco
HOST := $(shell hostname)
USR  := $(shell whoami)
DIR_SUFFIX := $(shell date +%d%m%y_%H%M)
DIR := /scratch/$(USR)/build_$(HOST)/build_$(DIR_SUFFIX)
AOCX_DIR := $(DIR)/$(O)



ifneq ($(K),)
STAT := $(shell rm -rf $(AOCX_DIR)) 
#STAT := $(shell mkdir -p $(AOCX_DIR)) 
#STAT := test -e $(DIR)/$(AOCX_OUT) || $(shell touch $(DIR)/$(AOCX_OUT) $(DIR)/$(AOCO_OUT))
##STAT := $(shell ln -s $(DIR)/$(AOCX_OUT) . )
##STAT := $(shell ln -s $(DIR)/$(AOCO_OUT) . )
#STAT := $(shell ln -s $(AOCX_DIR)        . )
endif

#ifeq "$(wildcard $(AOCX_DIR) )" "" 
##@echo $(AOCX_DIR)
#endif

############################################################################################################
# Target, Include files, and CPP/C files
############################################################################################################
emulate : CPPFLAGS += -DEMULATOR 

# Make it all!
all : $(TARGET_DIR)/$(TARGET)

emulate : $(AOCX_OUT) $(TARGET_DIR)/$(TARGET)
	rm -f run.log
	env CL_CONTEXT_EMULATOR_DEVICE_ALTERA=1 $(TARGET_DIR)/$(TARGET) -x=../$(O) -m=$(M) -n=$(N) -p=$(P) | tee run.log

full_compile: $(KERNEL)	
	$(AOC) -g -v --report --profile $(KERNEL) -o $(AOCX_OUT)  \
	-I $(TIMER_PATH)/src/c -l timer.aoclib -L $(TIMER_PATH)

#emulate
$(AOCX_OUT) : Makefile $(wildcard src/*.cl src/*.h)
	@echo "-I- : Emulating $(KERNEL)"
	$(AOC) -v -march=emulator $(KERNEL) -o $(AOCX_OUT) -DEMULATOR \
	-I $(TIMER_PATH)/src/c -l timer.aoclib -L $(TIMER_PATH)

# Host executable target.
$(TARGET_DIR)/$(TARGET) : Makefile $(SRCS) $(INCS)
	@[ -d $(TARGET_DIR) ] || mkdir $(TARGET_DIR)
	$(ECHO)$(CXX) $(CPPFLAGS) $(CXXFLAGS) -fPIC $(foreach D,$(INC_DIRS),-I$D) \
			$(AOCL_COMPILE_CONFIG) $(SRCS) $(AOCL_LINK_CONFIG) \
			$(foreach D,$(LIB_DIRS),-L$D) \
			$(foreach L,$(LIBS),-l$L) \
			-o $(TARGET_DIR)/$(TARGET)

# Standard make targets
clean :
	$(ECHO)rm -f $(TARGET_DIR)/$(TARGET)

run : $(TARGET_DIR)/$(TARGET)
	rm -f run.log
	$(TARGET_DIR)/$(TARGET) -x=../$(O) -m=$(M) -n=$(N) -p=$(P) | tee run.log

.PHONY : all clean emulate run
