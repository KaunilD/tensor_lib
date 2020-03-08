// tensor_lib.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <array>
#include <memory>
#include <cassert>
#include <iostream>
#include "cuda_runtime.h"

template<int>
class host_tensor;

template<int DIMS>
class tensor {


protected:
	// total number of floats held by this tensor
	uint32_t m_num_elements;

	// RAII on the m_data array
	std::shared_ptr<float> m_data_ptr;
	float* m_data;

	virtual void allocate_data() = 0;

	__host__ __device__  float* get() const;

	void set_n_elems();
public:
	// number of elements in each dimension
	const std::array<int, DIMS> m_size;

	tensor(const std::array<int, DIMS> t_size);
	
	__host__ __device__ uint32_t get_n_elems() const;
	
	__host__ __device__ __inline__ float& at(size_t x);

};
