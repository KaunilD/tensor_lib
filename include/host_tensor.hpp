#include "tensor.hpp"

template<int DIMS>
class host_tensor : public tensor<DIMS> {
protected:
	virtual void allocate_data();
public:

	host_tensor(const std::array<int, DIMS> );
};
