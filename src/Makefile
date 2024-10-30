# Name of the compiler
CC = gcc

# Compiler flags
CFLAGS = -Wall -Wextra -g

# The target executable
TARGET = main

xor: xor.o
	$(CC) $(CFLAGS) xor.o -o xor -lm

xor.o: xor.c
	$(CC) $(CFLAGS) -c xor.c -lm
# Rule to link and create the executable

main: main.o
	$(CC) $(CFLAGS) -o $(TARGET) main.o -lm

# Rule to compile the .c file into a .o file
main.o: main.c
	$(CC) $(CFLAGS) -c main.c -lm


# Clean up object files and the executable
clean:
	rm -f $(TARGET) main.o