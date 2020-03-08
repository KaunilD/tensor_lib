#include "host_tensor.hpp"

struct host_deletor {
	void operator()(float * ptr) const {
		delete[] ptr;
	}
};

template<int DIMS>
void host_tensor<DIMS>::allocate_data() {
	this->m_data = new float[this->get_n_elems()];
	this->m_data_ptr = std::shared_ptr<float>(this->m_data, host_deletor());
};

template<int DIMS>
host_tensor<DIMS>::host_tensor(const std::array<int, DIMS> t_size) : tensor<DIMS>(t_size) {
	allocate_data();
};

template class host_tensor<1>;