SOURCE=../src/test.cpp
INCLUDES=../include/
CC=g++
CXXFLAGS= -Wall -g -pipe -O2 -fPIC

all: sgd_lr

sgd_lr: $(SOURCE)
	$(CC) $(CXXFLAGS) -I $(INCLUDES) $(SOURCE) -o sgd_lr

clean:
	rm -f sgd_lr

