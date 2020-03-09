#include <iostream>
#include "host_tensor.hpp"
#include "device_tensor.hpp"

#define LOG(x) {std::cout << x << std::endl;}

using namespace std;

int main()
{
	LOG("Hello CMake")
	device_tensor<1> a({1024});
	device_tensor<1> b(a);
	LOG(b.get_n_elems())
	return 0;
}