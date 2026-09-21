#!/usr/bin/env bash
set -euo pipefail
mode="${1:?usage: build-native.sh direct|embedded}"
: "${ANDROID_NDK:?}" "${OPENCL_INCLUDE_DIR:?}" "${OPENCL_LIBRARY:?}"
case "$(uname -s)" in
  Darwin) host=darwin-x86_64 ;;
  Linux) host=linux-x86_64 ;;
  *) exit 1 ;;
esac
case "$mode" in
  direct) flags=(-DDIRECT_OPENCL) ;;
  embedded) flags=("$OPENCL_LIBRARY") ;;
  *) echo "Unknown mode: $mode" >&2; exit 1 ;;
esac
app_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$app_dir/src/main/jniLibs/arm64-v8a"
"$ANDROID_NDK/toolchains/llvm/prebuilt/$host/bin/aarch64-linux-android28-clang" \
  -shared -fPIC -O2 -g -Wall -Wextra -Werror -fvisibility=hidden \
  -I"$OPENCL_INCLUDE_DIR" "$app_dir/probe.c" "${flags[@]}" \
  -Wl,--exclude-libs,ALL -Wl,--no-undefined -ldl -llog \
  -o "$app_dir/src/main/jniLibs/arm64-v8a/libopencl_probe.so"
