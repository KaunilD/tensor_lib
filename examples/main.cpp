#include <iostream>
#include "host_tensor.hpp"
#include "device_tensor.hpp"

#define LOG(x) std::cout << x << std::endl

using namespace std;

int main()
{
	LOG("Hello Cmake.");
	host_tensor<1> a({ 1024 }, 2.0f);
	LOG(a.at(1023));
	return 0;
}