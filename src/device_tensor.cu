#include "device_tensor.hpp"


struct cuda_deletor {
	void operator()(float* p) const {
		cudaFree(p);
	}
};

template<int DIMS>
void device_tensor<DIMS>::allocate_data() {
	cudaMalloc(&(this->m_data), this->get_n_elems() * sizeof(float));
	this->m_data_ptr = std::shared_ptr<float>(this->m_data, cuda_deletor());
};

template <int DIMS>
device_tensor<DIMS>::device_tensor(const std::array<int, DIMS> t_size) :tensor<DIMS>(t_size) {
	allocate_data();
};

template class device_tensor<1>;