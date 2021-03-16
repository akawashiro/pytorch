#include <ATen/detail/CUDAHooksInterface.h>

#include <c10/util/Exception.h>

#include <iostream>
#include <cstddef>
#include <memory>
#include <mutex>

namespace at {
namespace detail {

// NB: We purposely leak the CUDA hooks object.  This is because under some
// situations, we may need to reference the CUDA hooks while running destructors
// of objects which were constructed *prior* to the first invocation of
// getCUDAHooks.  The example which precipitated this change was the fused
// kernel cache in the JIT.  The kernel cache is a global variable which caches
// both CPU and CUDA kernels; CUDA kernels must interact with CUDA hooks on
// destruction.  Because the kernel cache handles CPU kernels too, it can be
// constructed before we initialize CUDA; if it contains CUDA kernels at program
// destruction time, you will destruct the CUDA kernels after CUDA hooks has
// been unloaded.  In principle, we could have also fixed the kernel cache store
// CUDA kernels in a separate global variable, but this solution is much
// simpler.
//
// CUDAHooks doesn't actually contain any data, so leaking it is very benign;
// you're probably losing only a word (the vptr in the allocated object.)
static CUDAHooksInterface* cuda_hooks = nullptr;

const CUDAHooksInterface& getCUDAHooks() {
  // NB: The once_flag here implies that if you try to call any CUDA
  // functionality before libATen_cuda.so is loaded, CUDA is permanently
  // disabled for that copy of ATen.  In principle, we can relax this
  // restriction, but you might have to fix some code.  See getVariableHooks()
  // for an example where we relax this restriction (but if you try to avoid
  // needing a lock, be careful; it doesn't look like Registry.h is thread
  // safe...)
  std::cerr << __FILE__ << ":" << __LINE__ << std::endl;

  auto c2 = new CUDAHooksInterface();
  std::cerr << __FILE__ << ":" << __LINE__ <<  " typeid(*c2).name() = "  << typeid(*c2).name() << std::endl;
  delete c2;

  static std::once_flag once;
  std::cerr << __FILE__ << ":" << __LINE__ << std::endl;
  
  std::cerr << __FILE__ << ":" << __LINE__ << " cuda_hooks  = " << std::hex << cuda_hooks << std::dec << std::endl;

  std::call_once(once, [] {
    std::cerr << __FILE__ << ":" << __LINE__ << " call_once in CUDAHookInterface.cpp" << std::endl;
    auto p = CUDAHooksRegistry();
    std::cerr << __FILE__ << ":" << __LINE__ << " call_once in CUDAHookInterface.cpp" << std::endl;
    auto c = p->Create("CUDAHooks", CUDAHooksArgs{});
    std::cerr << __FILE__ << ":" << __LINE__ << " call_once in CUDAHookInterface.cpp" << " typeid(c) = " << typeid(c).name() << std::endl;
    cuda_hooks = c.release();
    std::cerr << __FILE__ << ":" << __LINE__ << " cuda_hooks  = " << std::hex << cuda_hooks << std::dec << std::endl;
    std::cerr << __FILE__ << ":" << __LINE__ << " call_once in CUDAHookInterface.cpp" << std::endl;
    if (!cuda_hooks) {
    std::cerr << __FILE__ << ":" << __LINE__ << " call_once in CUDAHookInterface.cpp" << std::endl;
      // cuda_hooks = new CUDAHooksInterface();
    std::cerr << __FILE__ << ":" << __LINE__ << " call_once in CUDAHookInterface.cpp" << std::endl;
    }
    std::cerr << __FILE__ << ":" << __LINE__ << " call_once in CUDAHookInterface.cpp" << std::endl;
    std::cerr << __FILE__ << ":" << __LINE__ << " cuda_hooks  = " << std::hex << cuda_hooks << std::dec << std::endl;
char str[1024];
  FILE* fp = fopen("/proc/self/maps", "r");
  while ((fgets(str, 256, fp)) != NULL) {
      printf("%s", str);
  }
  fclose(fp);
    std::cerr << __FILE__ << ":" << __LINE__ << " call_once in CUDAHookInterface.cpp" << " typeid(*cuda_hooks).name() = "  << typeid(*cuda_hooks).name() << std::endl;
  });
  std::cerr << __FILE__ << ":" << __LINE__ << std::endl;
  std::cerr << __FILE__ << ":" << __LINE__ << " cuda_hooks  = " << std::hex << cuda_hooks << std::dec << std::endl;
  
  std::cerr  << __FILE__ << ":" << __LINE__ << " " << cuda_hooks->compiledWithCuDNN() << std::endl;
  return *cuda_hooks;
}
} // namespace detail

C10_DEFINE_REGISTRY(CUDAHooksRegistry, CUDAHooksInterface, CUDAHooksArgs)

} // namespace at
