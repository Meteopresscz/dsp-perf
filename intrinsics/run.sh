#!/bin/bash

CC=gcc

CFLAGS="
-O3
-fno-common 
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

rm pol?.c.bin pol?.py.bin

seq 100000 | head -c 240000 > input

echo "Testing NumPy implementation"
./process.py

for i in NAIVE UNROLL X64 SSE AVX; do
  echo -e "\n\n\n"
  echo "Compiling $i implementation"
  rm -f process
  $CC $CFLAGS -D$i process.c -g -ggdb3 -lvolk -o process
  echo "Testing $i implementation"
  ./process
  sha256sum -c < SHA256SUMS
done
