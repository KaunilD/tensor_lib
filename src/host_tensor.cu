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
void host_tensor<DIMS>::fill(float f) {
	for (size_t i = 0; i < this->get_n_elems(); i++) {
		this->get()[i] = f;
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

template<int DIMS>
host_tensor<DIMS>::host_tensor(const std::array<int, DIMS> t_size, float t_value) : tensor<DIMS>(t_size) {
	allocate_data();
	fill(t_value);
};

template<int DIMS>
host_tensor<DIMS>::host_tensor(const host_tensor<1>& t_hostTensor, bool copy) : tensor<DIMS>(t_hostTensor.m_size) {
	allocate_data();
	if (copy) {
		this->copy(t_hostTensor);
	}
};

template<int DIMS>
host_tensor<DIMS>::host_tensor(const device_tensor<1>& t_deviceTensor, bool copy) : tensor<DIMS>(t_deviceTensor.m_size) {
	allocate_data();
	if (copy) {
		this->copy(t_deviceTensor);
	}
};


template <int DIMS>
void host_tensor<DIMS>::copy(const host_tensor<DIMS>& t_hostTensor) {
	assert(this->get_n_elems() == t_hostTensor.get_n_elems());
	for (size_t i = 0; i < this->get_n_elems(); i++) {
		this->get()[i] = t_hostTensor.get()[i];
	}
};

template <int DIMS>
void host_tensor<DIMS>::copy(const device_tensor<DIMS>& t_deviceTensor) {
	assert(this->get_n_elems() == t_deviceTensor.get_n_elems());
	cudaMemcpy(this->get(), t_deviceTensor.get(), this->get_n_elems() * sizeof(float), cudaMemcpyDeviceToHost);
};



template class host_tensor<1>;