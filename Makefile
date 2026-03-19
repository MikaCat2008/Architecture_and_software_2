SRC_DIR = src
BIN_DIR = bin
BUILD_DIR = build

CXX = cl
CXXFLAGS = /std:c++20 /Zi /Od /EHsc /utf-8 /FS /Fd"$(BUILD_DIR)\\"
VCVARS = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

SRCS = main.cpp
OBJS = $(BUILD_DIR)\main.obj

TARGET = $(BIN_DIR)\main.exe

all: $(BUILD_DIR) $(BIN_DIR) $(TARGET)

$(TARGET): $(OBJS)
	call $(VCVARS) && link $(OBJS) /OUT:$(TARGET) /DEBUG

$(BUILD_DIR)\main.obj: $(SRC_DIR)\main.cpp
	call $(VCVARS) && $(CXX) $(CXXFLAGS) /c $(SRC_DIR)\main.cpp /Fo$(BUILD_DIR)\main.obj

$(BUILD_DIR):
	if not exist $(BUILD_DIR) mkdir $(BUILD_DIR)

$(BIN_DIR):
	if not exist $(BIN_DIR) mkdir $(BIN_DIR)
