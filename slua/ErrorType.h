/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <string>

namespace slua
{
	//throw inside a check to provide a custom message
	struct Error { std::string msg; };
}