#!/bin/bash

set -u

readonly CUDA_REQUIRED_VERSION="11.8"
readonly CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda-11.8}"
readonly CUDA_COMPILER="$CUDA_ROOT/bin/nvcc"
readonly C_COMPILER="${CARTGS_C_COMPILER:-/usr/bin/gcc}"
readonly CXX_COMPILER="${CARTGS_CXX_COMPILER:-/usr/bin/g++}"

errors=0
declare -a missing_packages=()

pass() {
      printf '[OK]      %s\n' "$1"
}

fail() {
      printf '[MISSING] %s\n' "$1" >&2
      errors=$((errors + 1))
}

add_package() {
      local package="$1"
      local existing
      for existing in "${missing_packages[@]:-}"; do
            if [ "$existing" = "$package" ]; then
                  return
            fi
      done
      missing_packages+=("$package")
}

check_command() {
      local command_name="$1"
      local apt_package="$2"
      if command -v "$command_name" >/dev/null 2>&1; then
            pass "$command_name: $(command -v "$command_name")"
      else
            fail "command '$command_name' (apt package: $apt_package)"
            add_package "$apt_package"
      fi
}

check_apt_package() {
      local package="$1"
      local purpose="$2"
      local required_file_pattern="${3:-}"
      if dpkg-query -W -f='${Status}' "$package" 2>/dev/null \
            | grep -q '^install ok installed$'; then
            if [ -n "$required_file_pattern" ] \
                  && ! dpkg -L "$package" 2>/dev/null \
                        | grep -Eq "$required_file_pattern"; then
                  fail "$package is installed but its CMake config is missing ($purpose)"
                  add_package "$package"
            else
                  pass "$package ($purpose)"
            fi
      else
            fail "$package ($purpose)"
            add_package "$package"
      fi
}

printf '%s\n' 'Checking CaRtGS server build dependencies...'

check_command ninja ninja-build
check_command pkg-config pkg-config
check_command wget wget
check_command unzip unzip
check_command tar tar

if command -v cmake >/dev/null 2>&1; then
      cmake_version=$(cmake --version 2>/dev/null | awk 'NR == 1 { print $3 }')
      cmake_major=${cmake_version%%.*}
      if [ -n "$cmake_version" ] \
            && [ "$(printf '%s\n' 3.20 "$cmake_version" | sort -V | head -n 1)" = "3.20" ] \
            && [ "$cmake_major" -lt 4 ]; then
            pass "cmake $cmake_version (required: >= 3.20 and < 4)"
      else
            fail "usable CMake version >= 3.20 and < 4; install CMake 3.27.9"
      fi
else
      fail "cmake; install CMake 3.27.9"
fi

if [ -x "$CUDA_COMPILER" ]; then
      if "$CUDA_COMPILER" --version \
            | grep -Eq "release ${CUDA_REQUIRED_VERSION}([,.]|$)"; then
            pass "CUDA $CUDA_REQUIRED_VERSION: $CUDA_COMPILER"
      else
            fail "CUDA $CUDA_REQUIRED_VERSION at $CUDA_COMPILER"
      fi
else
      fail "CUDA compiler: $CUDA_COMPILER"
fi

if [ -x "$C_COMPILER" ] && [ -x "$CXX_COMPILER" ]; then
      c_version=$("$C_COMPILER" -dumpfullversion -dumpversion 2>/dev/null)
      cxx_version=$("$CXX_COMPILER" -dumpfullversion -dumpversion 2>/dev/null)
      if [[ "$c_version" =~ ^9\. ]] && [[ "$cxx_version" =~ ^9\. ]]; then
            pass "GCC/G++ 9: $C_COMPILER / $CXX_COMPILER"
      else
            fail "GCC/G++ 9 (found: ${c_version:-unknown} / ${cxx_version:-unknown})"
            add_package gcc-9
            add_package g++-9
      fi

      if printf '#include <vector>\nint main() { return 0; }\n' \
            | "$CXX_COMPILER" -x c++ -fsyntax-only - >/dev/null 2>&1; then
            pass "C++ standard library headers"
      else
            fail "C++ standard library headers (apt package: build-essential)"
            add_package build-essential
      fi
else
      fail "GCC/G++ compiler paths: $C_COMPILER / $CXX_COMPILER"
      add_package gcc-9
      add_package g++-9
      add_package build-essential
fi

if command -v dpkg-query >/dev/null 2>&1; then
      check_apt_package libboost-serialization-dev 'Boost.Serialization'
      check_apt_package libssl-dev 'OpenSSL libcrypto'
      check_apt_package libglm-dev 'GLM headers and glmConfig.cmake' '/(glmConfig|glm-config)\.cmake$'
      check_apt_package libglfw3-dev 'GLFW library and glfw3Config.cmake' '/(glfw3Config|glfw3-config)\.cmake$'
      check_apt_package libopengl-dev 'OpenGL development library'
      check_apt_package libgl1-mesa-dev 'OpenGL GL development files'
else
      fail "dpkg-query; this preflight currently supports Ubuntu/Debian servers"
fi

printf '%s\n' '[MANAGED] Eigen 3.3.7, JsonCpp 1.9.5, LibTorch 2.3.1+cu118, and OpenCV are installed locally by build.sh.'
printf '%s\n' '[OPTIONAL] librealsense2-dev is needed only for the RealSense executable.'

if [ "$errors" -ne 0 ]; then
      printf '\nFound %d missing or invalid requirement(s).\n' "$errors" >&2
      if [ "${#missing_packages[@]}" -gt 0 ]; then
            printf '%s\n' 'Install all missing apt packages with:' >&2
            printf 'sudo apt-get update && sudo apt-get install -y' >&2
            printf ' %q' "${missing_packages[@]}" >&2
            printf '\n' >&2
      fi
      printf '%s\n' 'For CMake, use: python3 -m pip install --user --force-reinstall cmake==3.27.9' >&2
      exit 1
fi

printf '%s\n' 'All required server dependencies are available.'
