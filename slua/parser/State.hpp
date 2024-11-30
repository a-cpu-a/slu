/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <string>
#include <span>

//https://www.lua.org/manual/5.4/manual.html
//https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form
//https://www.sciencedirect.com/topics/computer-science/backus-naur-form

namespace slua
{
	struct Position
	{
		size_t line;
		size_t index;
	};

	struct Input
	{
		std::span<const uint8_t> text;
		size_t idx;
	};
}