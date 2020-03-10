#include <iostream>
#include "host_tensor.hpp"
#include "device_tensor.hpp"

#define LOG(x) std::cout << x << std::endl

using namespace std;


struct ex {
	host_tensor<1> a;
	ex(int b) :a({b}, 1.0f) {}
};


int main()
{
	LOG("Hello Cmake.");
	host_tensor<1> a({10}, 2.0f);
	host_tensor<1> b(a);
	LOG(b.at(4));
	return 0;
}