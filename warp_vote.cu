#include <iostream>
#include <iomanip>
#include <bitset>
#include "cuda_ptr.cuh"

__global__ void test_any(int* a,
                         int* b,
                         int* c) {
  const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  a[tid] = __any(threadIdx.x == 16);
  b[tid] = __any(threadIdx.x == 128);
  if (threadIdx.x == 10) c[tid] = __any(threadIdx.x == 16);
  // if (threadIdx.x == 15 || threadIdx.x == 16) c[tid] = __any(threadIdx.x == 16);
  // if (threadIdx.x == 10 || threadIdx.x == 16) c[tid] = __any(threadIdx.x == 16);
}

__global__ void test_all(int* a,
                         int* b,
                         int* c) {
  const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  a[tid] = __all(threadIdx.x == 16);
  b[tid] = __all(b[tid] == -1);
  if (threadIdx.x == 10) c[tid] = __all(threadIdx.x == 16);
}

__global__ void test_ballot(int* a,
                            int* b,
                            int* c) {
  const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  a[tid] = __ballot(threadIdx.x == 16);
  b[tid] = __ballot(threadIdx.x % 2 == 0);
  if (threadIdx.x == 16) c[tid] = __ballot(threadIdx.x == 16);
}

int main() {
  const int tb_size = 64;
  const int grid_size = 2;
  const int array_size = tb_size * grid_size;

  cuda_ptr<int> a, b, c;
  a.allocate(array_size);
  b.allocate(array_size);
  c.allocate(array_size);
  a.set_val(-1);
  b.set_val(-1);
  c.set_val(-1);

  a.host2dev();
  b.host2dev();
  c.host2dev();
#ifdef TEST_ANY
  std::cerr << "TEST_ANY\n";
  test_any<<<grid_size, tb_size>>>(a, b, c);
#elif TEST_ALL
  std::cerr << "TEST_ALL\n";
  test_all<<<grid_size, tb_size>>>(a, b, c);
#elif TEST_BALLOT
  std::cerr << "TEST_BALLOT\n";
  test_ballot<<<grid_size, tb_size>>>(a, b, c);
#endif
  a.dev2host();
  b.dev2host();
  c.dev2host();
  
  std::cout << "threadIdx.x i a b c\n";

  for (int i = 0; i < array_size; i++) {
#ifdef TEST_BALLOT
    std::cout << i % tb_size << " " << i  << " " <<
      static_cast<std::bitset<32> >(a[i]) << " " <<
      static_cast<std::bitset<32> >(b[i]) << " " <<
      static_cast<std::bitset<32> >(c[i]) << "\n";
#else
    std::cout << i % tb_size << " " << i << " " << a[i] << " " << b[i] << " " << c[i] << "\n";
#endif
  }
}
