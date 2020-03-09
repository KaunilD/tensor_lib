#ifndef __HOST_TENSOR_H__
#define __HOST_TENSOR_H__

#include "tensor.hpp"
#include "device_tensor.hpp"

template<int DIMS>
class host_tensor : public tensor<DIMS> {

	friend device_tensor<DIMS>;
protected:

	virtual void allocate_data();
	virtual void copy(const host_tensor<DIMS>&);
	virtual void copy(const device_tensor<DIMS>&);
public:

	host_tensor(const std::array<int, DIMS>);
	host_tensor(const std::array<int, DIMS>, bool);

	host_tensor(const device_tensor<1>&, bool);
	host_tensor(const host_tensor<1>&, bool);
	void fill_random();

};
#endif