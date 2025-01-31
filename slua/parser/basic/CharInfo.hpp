/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <cstdint>

namespace sluaParse
{
	constexpr bool isDigitChar(const char c)
	{
		return c >= '0' && c <= '9';
	}
	constexpr bool isHexDigitChar(const char c)
	{
		return isDigitChar(c) || (c >= 'A' && c <= 'F') || (c >= 'a' && c <= 'f');
	}
	constexpr bool isValidNameStartChar(const char c)
	{// Check if the character is in the range of 'A' to 'Z' or 'a' to 'z', or '_'

		return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_';
	}
	constexpr bool isValidNameChar(const char c)
	{// Check if the character is in the range of '0' to '9', 'A' to 'Z' or 'a' to 'z', or '_'
		return isDigitChar(c) || isValidNameStartChar(c);
	}
}