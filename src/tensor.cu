#include "tensor.hpp"

template<int DIMS>
__host__ __device__ 
float* tensor<DIMS>::get() const { 
	return m_data; 
};


template<int DIMS>
void tensor<DIMS>::set_n_elems() {
	m_num_elements += m_size[0];
};

template<int DIMS>
tensor<DIMS>::tensor(const std::array<int, DIMS> t_size) : m_size(t_size) {
	static_assert(DIMS == 1, "Only 1D Tensors supported!");
	set_n_elems();
};

template<int DIMS>
__host__ __device__ 
uint32_t tensor<DIMS>::get_n_elems() const {
	return m_num_elements; 
};

template<int DIMS>
__host__ __device__ 
float& tensor<DIMS>::at(size_t x) {
	return *(this->m_data + x);
};

template class tensor<1>;