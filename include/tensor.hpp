// tensor_lib.h : Include file for standard system include files,
// or project specific include files.
#ifndef __TENSOR_H__
#define __TENSOR_H__

#include <array>
#include <memory>
#include <cassert>
#include <iostream>
#include "cuda_runtime.h"

template<int>
class device_tensor;

template<int>
class host_tensor;

template<int DIMS>
class tensor {


protected:
	// total number of floats held by this tensor
	size_t m_num_elements{0};

	// RAII on the m_data array
	std::shared_ptr<float> m_data_ptr;
	float* m_data{nullptr};

	virtual void allocate_data() = 0;

	__host__ __device__  float* get() const;

	void set_n_elems();
public:
	// number of elements in each dimension
	const std::array<int, DIMS> m_size;

	tensor(const std::array<int, DIMS> t_size);
	
	__host__ __device__ size_t get_n_elems() const;
	
	/*	Accessors to return elements 
		stored in ROW MAJOR order
	*/
	/* 1D access: returns this[x] */
	__host__ __device__ __inline__ float& at(size_t x);
	/* 2D access: returns this[x][y] */
	__host__ __device__ __inline__ float& at(size_t x, size_t y);

	virtual void copy(const host_tensor<DIMS>&)		= 0;
	virtual void copy(const device_tensor<DIMS>&)	= 0;

};
#endif