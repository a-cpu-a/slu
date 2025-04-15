/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <string>
#include <vector>

namespace slua::lang
{
	//Might in the future also contain data about other stuff, like export control (crate,self,tests,...).
	using ExportData = bool;

	using ModPath = std::vector<std::string>;
}