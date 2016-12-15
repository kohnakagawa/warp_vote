#pragma once

#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <thrust/fill.h>
#include <thrust/device_ptr.h>
#include <thrust/device_new.h>
#include <helper_cuda.h>

template <typename T>
struct cuda_ptr {
  T* dev_ptr  = nullptr;
  T* host_ptr = nullptr;
  int size = -1;
  thrust::device_ptr<T> thrust_ptr;

  cuda_ptr<T>() {}
  ~cuda_ptr() {
    deallocate();
  }

  // disable copy constructor
  const cuda_ptr<T>& operator = (const cuda_ptr<T>& obj) = delete;
  cuda_ptr<T>(const cuda_ptr<T>& obj) = delete;

  cuda_ptr<T>& operator = (cuda_ptr<T>&& obj) noexcept {
    this->dev_ptr = obj.dev_ptr;
    this->host_ptr = obj.host_ptr;

    obj.dev_ptr = nullptr;
    obj.host_ptr = nullptr;

    return *this;
  }
  cuda_ptr<T>(cuda_ptr<T>&& obj) noexcept {
    *this = std::move(obj);
  }

  void allocate(const int size_) {
    size = size_;
    checkCudaErrors(cudaMalloc((void**)&dev_ptr, size * sizeof(T)));
    checkCudaErrors(cudaMallocHost((void**)&host_ptr, size * sizeof(T)));
    thrust_ptr = thrust::device_pointer_cast(dev_ptr);
  }

  void host2dev(const int beg, const int count) {
    checkCudaErrors(cudaMemcpy(dev_ptr + beg,
                               host_ptr + beg,
                               count * sizeof(T),
                               cudaMemcpyHostToDevice));
  }
  void host2dev() {this->host2dev(0, size);}
  void host2dev_async(const int beg, const int count, cudaStream_t& strm) {
    checkCudaErrors(cudaMemcpyAsync(dev_ptr + beg,
                                    host_ptr + beg,
                                    count * sizeof(T),
                                    cudaMemcpyHostToDevice,
                                    strm));
  }

  void dev2host(const int beg,  const int count) {
    checkCudaErrors(cudaMemcpy(host_ptr + beg,
                               dev_ptr + beg,
                               count * sizeof(T),
                               cudaMemcpyDeviceToHost));
  }

  void dev2host() {this->dev2host(0, size);}

  void dev2host_async(const int beg, const int count, cudaStream_t& strm) {
    checkCudaErrors(cudaMemcpyAsync(host_ptr + beg,
                                    dev_ptr + beg,
                                    count * sizeof(T),
                                    cudaMemcpyDeviceToHost,
                                    strm));
  }

  void set_val(const T val) {
    std::fill(host_ptr, host_ptr + size, val);
    thrust::fill(thrust_ptr, thrust_ptr + size, val);
  }

  void set_val(const int beg, const int count, const T val){
    T* end_ptr = host_ptr + beg + count;
    std::fill(host_ptr + beg, end_ptr, val);
    thrust::device_ptr<T> beg_ptr = thrust_ptr + beg;
    thrust::fill(beg_ptr, beg_ptr + count, val);
  }

  const T& operator [] (const int i) const {
    return host_ptr[i];
  }

  T& operator [] (const int i) {
    return host_ptr[i];
  }

  operator const T* () const {
    return dev_ptr;
  }

  operator T* () {
    return dev_ptr;
  }

private:
  void deallocate() {
    checkCudaErrors(cudaFree(dev_ptr));
    checkCudaErrors(cudaFreeHost(host_ptr));
  }
};
