#include <iostream>
#include "host_tensor.hpp"
using namespace std;

int main()
{
	cout << "Hello CMake." << endl;
	host_tensor<1> a({1024});
	return 0;
}