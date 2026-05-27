#!/bin/bash

BUILD_DIR="build"
if [ ! -d $BUILD_DIR ]; then
   mkdir $BUILD_DIR
fi

BUILD_TYPE="Release"
if [[ $1 = "Debug" ]]; then
   BUILD_TYPE=$1
fi

cd $BUILD_DIR
cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE ..
cmake --build . -- -j8
cd ..
