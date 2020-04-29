#include "tensor.hpp"

template<int DIMS>
__host__ __device__
float* tensor<DIMS>::get() const {
	return m_data;
};

template<int DIMS>
void tensor<DIMS>::set_n_elems() {
	m_num_elements = 1;
	for (int i = 0; i < m_size.size(); i++) {
		m_num_elements *= m_size[i];
	}
};

template<int DIMS>
tensor<DIMS>::tensor(const std::array<int, DIMS> t_size) : m_size(t_size) {
	static_assert(DIMS <= 3, "Tensor class only supports upto 3 dimensions.\n");
	static_assert(DIMS >= 1, "Tensor must be atleast 1 dimensional!\n");
	set_n_elems();
};

template<int DIMS>
__host__ __device__
size_t tensor<DIMS>::get_n_elems() const {
	return m_num_elements;
};

template<int DIMS>
__host__ __device__
float& tensor<DIMS>::at(size_t x) const {
	static_assert(DIMS == 1, "Trying to use 1D accessor on a non 1D-Tensor.\n");
	return *(this->m_data + x);
};

template<int DIMS>
__host__ __device__
float& tensor<DIMS>::at(size_t x, size_t y) const {
	static_assert(DIMS == 2, "Trying to use 2D accessor on a non 2D-Tensor.\n");
	return *(this->m_data + x * m_size[1] + y);
};

template<int DIMS>
__host__ __device__
float& tensor<DIMS>::at(size_t x, size_t y, size_t z) const {
	static_assert(DIMS == 3, "Trying to use 3D accessor on a non 3D-Tensor.\n");
	return *(this->m_data + x * m_size[1] * m_size[2] + y * m_size[2] + z);
};

template class tensor<1>;
template class tensor<2>;
template class tensor<3>;
