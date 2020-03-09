#ifndef __HOST_TENSOR_H__
#define __HOST_TENSOR_H__

#include "tensor.hpp"
#include "device_tensor.hpp"

template<int DIMS>
class host_tensor : public tensor<DIMS> {

	friend device_tensor<DIMS>;

	virtual void allocate_data();
public:

	host_tensor(const std::array<int, DIMS>);
	host_tensor(const std::array<int, DIMS>, bool);
	void fill_random();
};
#endif