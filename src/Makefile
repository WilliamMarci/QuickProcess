# Makefile for checkRootFiles

# 编译器和 ROOT 配置
CXX       := g++
ROOTCFLAGS:= $(shell root-config --cflags)
ROOTLIBS  := $(shell root-config --libs)

# 编译选项
CXXFLAGS  := -std=c++11 $(ROOTCFLAGS)
LDFLAGS   := $(ROOTLIBS)

# 源文件和目标文件
SRCS      := checkRootFiles.cpp
OBJS      := $(SRCS:.cpp=.o)
TARGET    := ../bin/checkRootFiles

.PHONY: all clean

all: $(TARGET)

# 链接生成可执行文件
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

