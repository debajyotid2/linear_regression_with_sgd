CC=g++
CFLAGS=-Wall -O3 -Werror
CATCH2FLAGS=-lCatch2Main -lCatch2
BLASFLAGS=-lopenblas

SRC_DIR=src
OBJ_DIR=obj
TEST_DIR=tests
SRC=$(wildcard $(SRC_DIR)/*.c)
OBJ=$(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRC))
TESTS=$(wildcard $(TEST_DIR)/*.c)
TEST_BINS=$(patsubst $(TEST_DIR)/%.c, $(TEST_DIR)/bin/%, $(TESTS))

BIN=run

all: $(BIN)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c $(SRC_DIR)/%.h
	$(CC) $(CFLAGS) -c $< -o $@ $(BLASFLAGS)

$(BIN): main.c $(OBJ)
	$(CC) $(CFLAGS) $^ -o $@ $(BLASFLAGS)

$(TEST_DIR)/bin/%: $(TEST_DIR)/%.c $(OBJ)
	$(CC) $(CFLAGS) $^ -o $@ $(CATCH2FLAGS) $(BLASFLAGS)

test: $(OBJ) $(TEST_BINS)
	for test in $(TEST_BINS); do $$test; done

clean:
	$(RM) $(BIN) 
	$(RM) $(OBJ)
	$(RM) $(TEST_BINS)
