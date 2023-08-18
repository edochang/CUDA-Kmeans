#CC = g++
#SRCS = $(wildcard ./src/*.cpp)
#INC = ./src/
#OPTS = -std=c++17 -Wall -Werror

CC = nvcc
CCFlag = -arch=native
SRCS = $(wildcard ./src/*.cpp) $(wildcard ./src/*.cu)
INC = ./src/
OPTS = -std=c++17

EXEC = bin/kmeans

all: clean compile

compile:
#	$(CC) -arch compute_86 $(SRCS) $(OPTS) -I $(INC) -o $(EXEC)
#	$(CC) -arch sm_86 $(SRCS) $(OPTS) -I $(INC) -o $(EXEC)
	$(CC) $(CCFlag) $(SRCS) $(OPTS) -I $(INC) -o $(EXEC)
#	$(CC) -arch all-major $(SRCS) $(OPTS) -I $(INC) -o $(EXEC)
#	$(CC) -arch all $(SRCS) $(OPTS) -I $(INC) -o $(EXEC)
#	$(CC) $(SRCS) $(OPTS) -I $(INC) -o $(EXEC)

clean:
	rm -f $(EXEC)