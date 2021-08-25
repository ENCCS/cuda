# -*- coding: utf-8 -*-

from sphinx_lesson.directives import _BaseCRDirective


class SignatureDirective(_BaseCRDirective):
    extra_classes = ["toggle-shown", "dropdown"]


class ParametersDirective(_BaseCRDirective):
    extra_classes = ["dropdown"]


class TypealongDirective(_BaseCRDirective):
    extra_classes = ["toggle-shown", "dropdown"]


DIRECTIVES = [SignatureDirective, ParametersDirective, TypealongDirective]

API_REFERENCES_FUNCTION = [
    ("cudaGetDeviceCount", "https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g18808e54893cfcaafefeab31a73cc55f"),
    ("cudaGetDeviceProperties", "https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g1bf9d625a931d657e08db2b4391170f0"),
    ("cudaMalloc", "https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g37d37965bfb4803b6d4e59ff26856356"),
    ("cudaFree", "https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1ga042655cbbf3408f01061652a075e094"),
    ("cudaMemcpy", "https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gc263dbe6574220cc776b45438fc351e8"),
    ("__syncthreads", "https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#synchronization-functions"),
    ("cudaGetLastError", "https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__ERROR.html#group__CUDART__ERROR_1g3529f94cb530a83a76613616782bd233"),
    ("cudaGetErrorString", "https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__ERROR.html#group__CUDART__ERROR_1g4bc9e35a618dfd0877c29c8ee45148f1"),
    ("atomicAdd", "https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomicadd"),
    ("cudaMemset", "https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gf7338650f7683c51ee26aadc6973c63a"),
    ("cudaMallocHost", "https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gab84100ae1fa1b12eaca660207ef585b"),
    ("cudaFreeHost", "https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g71c078689c17627566b2a91989184969"),
    ("cudaMemcpyAsync", "https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g85073372f776b4c4d5f89f7124b7bf79"),
    ("cudaDeviceSynchronize", "https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g10e20b05a95f638a4071a655503df25d"),
    ("cudaStreamCreate", "https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g6a3c4b819e6a994c26d0c4824a4c80da"),
    ("cudaEventCreate", "https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html#group__CUDART__EVENT_1g7c581e3613a2110ba4d4e7fd5c7da418"),
    ("cudaEventRecord", "https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html#group__CUDART__EVENT_1gf4fcb74343aa689f4159791967868446"),
    ("cudaStreamWaitEvent", "https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g7840e3984799941a61839de40413d1d9"),
]

API_REFERENCES_FUNCTION_TERM = """
.. |{command}| raw:: html

   <a class="reference internal" href="{htmlref}"><span class="xref std std-term"><code class="docutils literal notranslate">{command}(..)</code></span></a>
"""

API_REFERENCES_VARIABLE = [
    ("cudaError_t", "https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g3f51e3575c2178246db0a94a430e0038"),
    ("cudaSuccess", "https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g3f51e3575c2178246db0a94a430e0038"),
    ("cudaErrorInvalidDevice", "https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g3f51e3575c2178246db0a94a430e0038"),
    ("cudaDeviceProp", "https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html"),
    ("cudaMemcpyKind", "https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g18fa99055ee694244a270e4d5101e95b"),
    ("cudaMemcpyHostToHost", "https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g18fa99055ee694244a270e4d5101e95b"),
    ("cudaMemcpyHostToDevice", "https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g18fa99055ee694244a270e4d5101e95b"),
    ("cudaMemcpyDeviceToHost", "https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g18fa99055ee694244a270e4d5101e95b"),
    ("cudaMemcpyDeviceToDevice", "https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g18fa99055ee694244a270e4d5101e95b"),
    ("cudaMemcpyDefault", "https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g18fa99055ee694244a270e4d5101e95b"),
    ("__global__", "https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#global"),
    ("__device__", "https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-function-specifier"),
    ("__host__", "https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#host"),
    ("float3", "https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#built-in-vector-types"),
    ("uint3", "https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#built-in-vector-types"),
    ("dim3", "https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#dim3"),
    ("__shared__", "https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory"),
    ("cudaStream_t", "https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1ge15d9c8b7a240312b533d6122558085a"),
    ("cudaEvent_t", "https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gea2f543a9fc0e52fe4ae712920fd1247"),
]

API_REFERENCES_VARIABLE_TERM = """
.. |{command}| raw:: html

   <a class="reference internal" href="{htmlref}"><span class="xref std std-term"><code class="docutils literal notranslate">{command}</code></span></a>
"""


def cuda_glossary():
    api_references_function = "\n".join(
        (API_REFERENCES_FUNCTION_TERM.format(command=x[0], htmlref=x[1]) for x in API_REFERENCES_FUNCTION)
    )
    api_references_variable = "\n".join(
        (API_REFERENCES_VARIABLE_TERM.format(command=x[0], htmlref=x[1]) for x in API_REFERENCES_VARIABLE)
    )
    return api_references_variable + api_references_function
