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

	/* create a host_tensor with default values */
	host_tensor(const std::array<int, DIMS>);
	/* create a host_tensor with random values */
	host_tensor(const std::array<int, DIMS>, bool);
	/* create a host_tensor with specified value */
	host_tensor(const std::array<int, DIMS>, float );

	/* copy constructor from the device_tensor, 
		second param is copy: if false then creates 
		a tensor with the same dimensions as the const reference
	*/
	host_tensor(const device_tensor<1>&, bool);
	host_tensor(const host_tensor<1>&, bool);
	
	/* helpers */
	void fill_random();
	void fill(float );

};
#endif