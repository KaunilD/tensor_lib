#include "host_tensor.hpp"

struct host_deletor {
	void operator()(float * ptr) const {
		delete[] ptr;
	}
};

template<int DIMS>
void host_tensor<DIMS>::fill_random() {
	for (size_t i = 0; i < this->get_n_elems(); i++) {
		this->get()[i] = float(rand()) / float(RAND_MAX) * 2.0f - 1.0f;
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

template<int DIMS>
host_tensor<DIMS>::host_tensor(const std::array<int, DIMS> t_size, bool rand) : tensor<DIMS>(t_size) {
	allocate_data();
	if (rand) {
		fill_random();
	}
};


template class host_tensor<1>;