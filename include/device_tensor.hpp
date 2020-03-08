#include "tensor.hpp"

template <int DIMS>
class device_tensor : public tensor<DIMS> {
protected:
	virtual void allocate_data();
public:
	device_tensor(const std::array<int, DIMS> );
};

