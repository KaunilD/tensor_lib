#ifndef __DEVICE_TENSOR_H__
#define __DEVICE_TENSOR_H__

#include "tensor.hpp"
#include "host_tensor.hpp"

template <int DIMS>
class device_tensor : public tensor<DIMS> {
	friend host_tensor<DIMS>;

protected:
	virtual void allocate_data();
public:
	device_tensor(const std::array<int, DIMS>);
	
	device_tensor(const host_tensor<DIMS>&);
	device_tensor(const device_tensor<DIMS>&);
	device_tensor(const device_tensor<DIMS>& t_deviceTensor, bool copy);

	device_tensor<DIMS>& operator=(const device_tensor<DIMS>& t_deviceTensor);
	
	void copy(const host_tensor<DIMS>& );
	void copy(const device_tensor<DIMS>& );
};
#endif