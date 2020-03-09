#include <iostream>
#include "host_tensor.hpp"
#include "device_tensor.hpp"

#define LOG(x) std::cout << x << std::endl

using namespace std;


struct ex {
	host_tensor<1> a;
	ex(const host_tensor<1>& b):a(b) {}
};


int main()
{
	LOG("Hello Cmake.");
	host_tensor<1> a({10}, 2.0f);
	ex e(a);
	LOG(e.a.at(9));
	return 0;
}