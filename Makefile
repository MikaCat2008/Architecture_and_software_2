CC = clang++
OBJS = main.o
CFLAGS = -g -std=c++20 -mavx -mxsave -mavx2 -msse3
SRC_DIR = src
BIN_DIR = bin
TARGET = $(BIN_DIR)/main.exe

all: $(TARGET)

$(TARGET): $(addprefix $(SRC_DIR)/,$(OBJS)) | $(BIN_DIR)
	$(CC) $(addprefix $(SRC_DIR)/,$(OBJS)) -o $(TARGET)
	del /s "src\*.o"

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@
