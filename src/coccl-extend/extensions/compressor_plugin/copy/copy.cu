#include "compressor_plugin/coccl_compressor_plugin.h"

namespace {

coccl::Status copyAsync(const void* source, size_t bytes, void* destination,
                        size_t capacity, cudaStream_t stream) {
  if (source == nullptr || destination == nullptr || bytes > capacity) {
    return ncclInvalidArgument;
  }
  if (source == destination || bytes == 0) return ncclSuccess;
  return coccl::fromCuda(cudaMemcpyAsync(destination, source, bytes,
                                         cudaMemcpyDeviceToDevice, stream));
}

struct CopyCompressor {
  static constexpr bool kBytewiseLossless = true;

  static coccl::Status compress(const coccl::Input& input,
                                coccl::Output& output,
                                coccl::Context& context) {
    if (coccl::dataTypeSize(input.datatype()) == 0 || input.chunks() == 0 ||
        input.elements() % input.chunks() != 0 ||
        input.bytes() % input.chunks() != 0) {
      return ncclInvalidArgument;
    }
    coccl::Status result = copyAsync(input.data(), input.bytes(), output.data(),
                                     output.capacityBytes(), context.stream());
    if (result != ncclSuccess) return result;
    return output.commitBytes(input.bytes(), input.chunks());
  }

  static coccl::Status decompress(const coccl::Input& input,
                                  coccl::Output& output,
                                  coccl::Context& context) {
    size_t outputBytes = 0;
    if (input.datatype() != ncclInt8 || input.chunks() == 0 ||
        input.elements() != input.bytes() ||
        output.chunks() != input.chunks() ||
        output.elements() % output.chunks() != 0 ||
        !coccl::checkedMultiply(output.elements(),
                                coccl::dataTypeSize(output.datatype()),
                                &outputBytes) ||
        outputBytes != input.bytes()) {
      return ncclInvalidArgument;
    }
    return copyAsync(input.data(), input.bytes(), output.data(),
                     output.capacityBytes(), context.stream());
  }

  static coccl::Status encodedSizeBound(const coccl::Shape& input,
                                        size_t* encodedBytes,
                                        const coccl::SizeContext& context) {
    if (context.operation() != cocclCompressorOperationCompress) {
      return ncclInvalidUsage;
    }
    const size_t bytes = input.bytes();
    if (encodedBytes == nullptr || bytes == 0 || input.chunks() == 0 ||
        input.elements() % input.chunks() != 0 ||
        bytes % input.chunks() != 0) {
      return ncclInvalidArgument;
    }
    *encodedBytes = bytes;
    return ncclSuccess;
  }
};

}  // namespace

COCCL_REGISTER_COMPRESSOR("copy", CopyCompressor)
