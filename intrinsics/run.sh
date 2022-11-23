#!/bin/bash

CC=gcc

CFLAGS="
-O3
-fno-common
-fno-strict-aliasing 
-fno-omit-frame-pointer 
-std=gnu11 
-Wall 
-Wextra 
-Wduplicated-cond 
-Wduplicated-branches 
-Wlogical-op 
-Wrestrict 
-Wnull-dereference 
-Wjump-misses-init 
-Wdouble-promotion 
-Wshadow 
-Wformat=2 
-march=native"

rm -f pol?.c.bin

#seq 100000 | head -c 240000 > input

testone() {
  i=$1
  echo -e "\n\n\n"
  echo "Compiling $i implementation"
  rm -f process
  $CC $CFLAGS -D$i process.c -g -ggdb3 -o process
  echo "Testing $i implementation"
  ./process
  sha256sum -c < SHA256SUMS
}

# test only one implementation if it is given by a command-line argument
if [ $# -gt 0 ]; then
  testone $1
  exit 0
fi

rm -f pol?.py.bin

echo "Testing NumPy implementation"
./process.py

for i in NAIVE UNROLL X64 SSE AVX; do
  testone $i
done
