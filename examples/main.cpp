#include <iostream>
#include "host_tensor.hpp"
#include "device_tensor.hpp"

#define LOG(x) {std::cout << x << std::endl;}

using namespace std;

int main()
{
	std::array<int, 1> size{1024};
	host_tensor<1> a(size, true);
	return 0;
}