/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <string>
#include <vector>

import slu.comp.cfg;

namespace slu::comp
{
	struct CodeGenEntrypoint
	{
		std::string entryPointFile;//path to file that defined this entry-point, or empty
		std::string fileName;
	};
	//Could repr some js file, some wasm blob, a jar / class, or even some exe / dll.
	struct CompEntryPoint : CodeGenEntrypoint
	{
		std::vector<uint8_t> contents;
	};
	struct CompOutput
	{
		std::vector<CompEntryPoint> entryPoints;
		//TODO: info for lock file appending?
		//TODO: info for build cache files?
	};

	CompOutput compile(const CompCfg& cfg);
}