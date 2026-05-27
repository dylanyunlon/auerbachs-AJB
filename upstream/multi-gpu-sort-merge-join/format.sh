#!/bin/bash

find src -regex '.*\.\(h\|c\|hpp\|cpp\|cuh\|cu\)' -exec clang-format -style=file -i {} \;

yapf -i --recursive scripts/*
