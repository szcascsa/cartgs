#!/bin/bash

set -euo pipefail

workdir=$(cd -- "$(dirname "$0")" >/dev/null 2>&1 && pwd -P)
cd "$workdir"

case "${1:-}" in
      "") ;;
      --check) ;;
      *)
            echo "Usage: $0 [--check]" >&2
            exit 2
            ;;
esac

if ! bash "$workdir/scripts/check_server_dependencies.sh"; then
      exit 1
fi

if [ "${1:-}" = "--check" ]; then
      exit 0
fi

readonly CUDA_REQUIRED_VERSION="11.8"
readonly CUDA_ARCHITECTURE="86"
readonly CUDA_ARCH_BIN="8.6"
readonly CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda-11.8}"
readonly CUDA_COMPILER="$CUDA_ROOT/bin/nvcc"
readonly C_COMPILER="${CARTGS_C_COMPILER:-/usr/bin/gcc}"
readonly CXX_COMPILER="${CARTGS_CXX_COMPILER:-/usr/bin/g++}"

if [ ! -x "$CUDA_COMPILER" ]; then
      echo "CUDA $CUDA_REQUIRED_VERSION compiler not found: $CUDA_COMPILER" >&2
      exit 1
fi

if ! "$CUDA_COMPILER" --version | grep -Eq "release ${CUDA_REQUIRED_VERSION}([,.]|$)"; then
      echo "CaRtGS requires CUDA $CUDA_REQUIRED_VERSION exactly: $CUDA_COMPILER" >&2
      "$CUDA_COMPILER" --version >&2
      exit 1
fi

if [ ! -x "$C_COMPILER" ] || [ ! -x "$CXX_COMPILER" ]; then
      echo "GCC/G++ compiler not found: $C_COMPILER / $CXX_COMPILER" >&2
      exit 1
fi

readonly C_COMPILER_VERSION=$("$C_COMPILER" -dumpfullversion -dumpversion)
readonly CXX_COMPILER_VERSION=$("$CXX_COMPILER" -dumpfullversion -dumpversion)
if [[ ! "$C_COMPILER_VERSION" =~ ^9\. ]] || [[ ! "$CXX_COMPILER_VERSION" =~ ^9\. ]]; then
      echo "CaRtGS requires GCC/G++ 9; found $C_COMPILER_VERSION / $CXX_COMPILER_VERSION" >&2
      echo "Override with CARTGS_C_COMPILER and CARTGS_CXX_COMPILER if GCC 9 is installed elsewhere." >&2
      exit 1
fi

if ! printf '#include <vector>\nint main() { return 0; }\n' \
      | "$CXX_COMPILER" -x c++ -fsyntax-only -; then
      echo "G++ cannot find the C++ standard library headers: $CXX_COMPILER" >&2
      exit 1
fi

export CUDA_HOME="$CUDA_ROOT"
export CUDA_PATH="$CUDA_ROOT"
export CUDAHOSTCXX="$CXX_COMPILER"
export CC="$C_COMPILER"
export CXX="$CXX_COMPILER"
export PATH="$CUDA_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_ROOT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TORCH_CUDA_ARCH_LIST="$CUDA_ARCH_BIN"

readonly LIBTORCH_ROOT="$workdir/third_party/libtorch-cu118"
readonly LIBTORCH_DIR="$LIBTORCH_ROOT/libtorch"
readonly LIBTORCH_ARCHIVE="$workdir/libtorch-cxx11-abi-shared-with-deps-2.3.1+cu118.zip"
readonly EIGEN_VERSION="3.3.7"
readonly EIGEN_SOURCE_DIR="$workdir/third_party/eigen-$EIGEN_VERSION"
readonly EIGEN_ARCHIVE="$workdir/eigen-$EIGEN_VERSION.tar.gz"
readonly EIGEN_BUILD_DIR="$EIGEN_SOURCE_DIR/build-cmake"
readonly EIGEN_INSTALL_DIR="$workdir/third_party/install/eigen-$EIGEN_VERSION"
readonly EIGEN_CONFIG_DIR="$EIGEN_INSTALL_DIR/share/eigen3/cmake"
readonly JSONCPP_VERSION="1.9.5"
readonly JSONCPP_SOURCE_DIR="$workdir/third_party/jsoncpp-$JSONCPP_VERSION"
readonly JSONCPP_ARCHIVE="$workdir/jsoncpp-$JSONCPP_VERSION.tar.gz"
readonly JSONCPP_BUILD_DIR="$JSONCPP_SOURCE_DIR/build-cmake"
readonly JSONCPP_INSTALL_DIR="$workdir/third_party/install/jsoncpp-$JSONCPP_VERSION"
readonly JSONCPP_CONFIG_DIR="$JSONCPP_INSTALL_DIR/lib/cmake/jsoncpp"
readonly OPENCV_BUILD_DIR="$workdir/third_party/opencv/build-cuda118-sm86-gcc9"
readonly OPENCV_INSTALL_DIR="$workdir/third_party/install/opencv-cuda118-sm86-gcc9"
readonly DBOW2_BUILD_DIR="$workdir/third_party/ORB-SLAM3/Thirdparty/DBoW2/build-cuda118-sm86-gcc9"
readonly G2O_BUILD_DIR="$workdir/third_party/ORB-SLAM3/Thirdparty/g2o/build-cuda118-sm86-gcc9"
readonly SOPHUS_BUILD_DIR="$workdir/third_party/ORB-SLAM3/Thirdparty/Sophus/build-cuda118-sm86-gcc9"
readonly ORB_SLAM3_BUILD_DIR="$workdir/third_party/ORB-SLAM3/build-cuda118-sm86-gcc9"
readonly CARTGS_BUILD_DIR="$workdir/build-cuda118-sm86-gcc9"

# LibTorch 2.3.1 provides the optimizer APIs used by CaRtGS and an official
# CUDA 11.8 C++ distribution.
if [ ! -d "$LIBTORCH_DIR/share/cmake/Torch" ]; then
      echo "Downloading LibTorch 2.3.1 + cu118 ..."
      mkdir -p "$LIBTORCH_ROOT"
      wget -c \
            -O "$LIBTORCH_ARCHIVE" \
            "https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.3.1%2Bcu118.zip"
      unzip -q -o "$LIBTORCH_ARCHIVE" -d "$LIBTORCH_ROOT"
      rm -f "$LIBTORCH_ARCHIVE"
fi

# Eigen is header-only, but its installed CMake package is required by g2o,
# Sophus, ORB-SLAM3, and CaRtGS.
if [ ! -f "$EIGEN_CONFIG_DIR/Eigen3Config.cmake" ]; then
      if [ ! -f "$EIGEN_SOURCE_DIR/CMakeLists.txt" ]; then
            echo "Downloading Eigen $EIGEN_VERSION ..."
            wget -c \
                  -O "$EIGEN_ARCHIVE" \
                  "https://gitlab.com/libeigen/eigen/-/archive/$EIGEN_VERSION/eigen-$EIGEN_VERSION.tar.gz"
            tar -xzf "$EIGEN_ARCHIVE" -C "$workdir/third_party"
            rm -f "$EIGEN_ARCHIVE"
      fi

      echo "Installing Eigen $EIGEN_VERSION locally ..."
      cmake -S "$EIGEN_SOURCE_DIR" -B "$EIGEN_BUILD_DIR" -G Ninja \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_C_COMPILER="$C_COMPILER" \
            -DCMAKE_CXX_COMPILER="$CXX_COMPILER" \
            -DCMAKE_INSTALL_PREFIX="$EIGEN_INSTALL_DIR" \
            -DBUILD_TESTING=OFF
      cmake --install "$EIGEN_BUILD_DIR"
fi

if [ ! -f "$JSONCPP_CONFIG_DIR/jsoncppConfig.cmake" ]; then
      if [ ! -f "$JSONCPP_SOURCE_DIR/CMakeLists.txt" ]; then
            echo "Downloading JsonCpp $JSONCPP_VERSION ..."
            wget -c \
                  -O "$JSONCPP_ARCHIVE" \
                  "https://github.com/open-source-parsers/jsoncpp/archive/refs/tags/$JSONCPP_VERSION.tar.gz"
            tar -xzf "$JSONCPP_ARCHIVE" -C "$workdir/third_party"
            rm -f "$JSONCPP_ARCHIVE"
      fi

      echo "Building JsonCpp $JSONCPP_VERSION locally ..."
      cmake -S "$JSONCPP_SOURCE_DIR" -B "$JSONCPP_BUILD_DIR" -G Ninja \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_C_COMPILER="$C_COMPILER" \
            -DCMAKE_CXX_COMPILER="$CXX_COMPILER" \
            -DCMAKE_INSTALL_PREFIX="$JSONCPP_INSTALL_DIR" \
            -DCMAKE_INSTALL_LIBDIR=lib \
            -DJSONCPP_WITH_TESTS=OFF \
            -DJSONCPP_WITH_POST_BUILD_UNITTEST=OFF \
            -DJSONCPP_WITH_CMAKE_PACKAGE=ON \
            -DJSONCPP_WITH_PKGCONFIG_SUPPORT=OFF \
            -DBUILD_SHARED_LIBS=ON \
            -DBUILD_STATIC_LIBS=OFF
      cmake --build "$JSONCPP_BUILD_DIR"
      cmake --install "$JSONCPP_BUILD_DIR"
fi

# OpenCV: build only for the RTX 3090 (compute capability 8.6).
echo "Building OpenCV for CUDA $CUDA_REQUIRED_VERSION / sm_$CUDA_ARCHITECTURE ..."
cmake -S "$workdir/third_party/opencv" -B "$OPENCV_BUILD_DIR" -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER="$C_COMPILER" \
      -DCMAKE_CXX_COMPILER="$CXX_COMPILER" \
      -DCMAKE_CUDA_COMPILER="$CUDA_COMPILER" \
      -DCMAKE_CUDA_HOST_COMPILER="$CXX_COMPILER" \
      -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURE" \
      -DCUDA_HOST_COMPILER="$CXX_COMPILER" \
      -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_ROOT" \
      -DCUDA_ARCH_BIN="$CUDA_ARCH_BIN" \
      -DCUDA_ARCH_PTX= \
      -DWITH_CUDA=ON -DWITH_CUDNN=OFF \
      -DWITH_CUFFT=ON -DWITH_CUBLAS=ON -DWITH_NVCUVENC=ON \
      -DOPENCV_DNN_CUDA=OFF -DWITH_NVCUVID=ON \
      -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF \
      -DBUILD_EXAMPLES=OFF -DBUILD_opencv_apps=OFF \
      -DBUILD_TIFF=ON -DBUILD_ZLIB=ON -DBUILD_JASPER=ON -DBUILD_CCALIB=ON \
      -DBUILD_JPEG=ON -DWITH_FFMPEG=ON \
      -DOPENCV_EXTRA_MODULES_PATH="$workdir/third_party/opencv_contrib/modules" \
      -DCMAKE_INSTALL_PREFIX="$OPENCV_INSTALL_DIR"
cmake --build "$OPENCV_BUILD_DIR"
cmake --install "$OPENCV_BUILD_DIR"

# DBoW2
cmake -S "$workdir/third_party/ORB-SLAM3/Thirdparty/DBoW2" \
      -B "$DBOW2_BUILD_DIR" -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER="$C_COMPILER" \
      -DCMAKE_CXX_COMPILER="$CXX_COMPILER" \
      -DOpenCV_DIR="$OPENCV_INSTALL_DIR/lib/cmake/opencv4"
cmake --build "$DBOW2_BUILD_DIR"

# g2o
cmake -S "$workdir/third_party/ORB-SLAM3/Thirdparty/g2o" \
      -B "$G2O_BUILD_DIR" -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER="$C_COMPILER" \
      -DCMAKE_CXX_COMPILER="$CXX_COMPILER" \
      -DEigen3_DIR="$EIGEN_CONFIG_DIR"
cmake --build "$G2O_BUILD_DIR"

# Sophus is header-only for CaRtGS; its tests and examples are unnecessary.
cmake -S "$workdir/third_party/ORB-SLAM3/Thirdparty/Sophus" \
      -B "$SOPHUS_BUILD_DIR" -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER="$C_COMPILER" \
      -DCMAKE_CXX_COMPILER="$CXX_COMPILER" \
      -DEigen3_DIR="$EIGEN_CONFIG_DIR" \
      -DBUILD_TESTS=OFF \
      -DBUILD_EXAMPLES=OFF
cmake --build "$SOPHUS_BUILD_DIR"

# ORB-SLAM3
echo "Uncompress vocabulary ..."
tar -xf "$workdir/third_party/ORB-SLAM3/Vocabulary/ORBvoc.txt.tar.gz" \
    -C "$workdir/third_party/ORB-SLAM3/Vocabulary"

cmake -S "$workdir/third_party/ORB-SLAM3" \
      -B "$ORB_SLAM3_BUILD_DIR" -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER="$C_COMPILER" \
      -DCMAKE_CXX_COMPILER="$CXX_COMPILER" \
      -DEigen3_DIR="$EIGEN_CONFIG_DIR" \
      -DOpenCV_DIR="$OPENCV_INSTALL_DIR/lib/cmake/opencv4"
cmake --build "$ORB_SLAM3_BUILD_DIR"

# CaRtGS
echo "Building CaRtGS for CUDA $CUDA_REQUIRED_VERSION / sm_$CUDA_ARCHITECTURE ..."
cmake -S "$workdir" -B "$CARTGS_BUILD_DIR" -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_COMPILER="$CXX_COMPILER" \
      -DCMAKE_CUDA_COMPILER="$CUDA_COMPILER" \
      -DCMAKE_CUDA_HOST_COMPILER="$CXX_COMPILER" \
      -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURE" \
      -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_ROOT" \
      -DEigen3_DIR="$EIGEN_CONFIG_DIR" \
      -Djsoncpp_DIR="$JSONCPP_CONFIG_DIR" \
      -DTorch_DIR="$LIBTORCH_DIR/share/cmake/Torch" \
      -DOpenCV_DIR="$OPENCV_INSTALL_DIR/lib/cmake/opencv4"
cmake --build "$CARTGS_BUILD_DIR"
