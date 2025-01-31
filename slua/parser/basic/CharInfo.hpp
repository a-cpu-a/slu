/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <cstdint>

namespace sluaParse
{
	constexpr uint8_t CAPITAL_BIT = 'x' - 'X';
	static_assert('x' - 'X' == 32);//is simple bit flip?

	constexpr char toLowerCase(const char c)
	{
		return c | CAPITAL_BIT;
	}

	constexpr bool isLowerCaseEqual(const char charToCheck,const char c)
	{
		return toLowerCase(charToCheck)==c;
	}
	constexpr bool isDigitChar(const char c)
	{
		return c >= '0' && c <= '9';
	}
	constexpr bool isAlpha(const char c)
	{
		return (c >= 'A' && c <= 'F') || (c >= 'a' && c <= 'f');
	}
	constexpr bool isHexDigitChar(const char c)
	{
		return isDigitChar(c) || isAlpha(c);
	}
	constexpr bool isValidNameStartChar(const char c)
	{// Check if the character is in the range of 'A' to 'Z' or 'a' to 'z', or '_'

		return isAlpha(c) || c == '_';
	}
	constexpr bool isValidNameChar(const char c)
	{// Check if the character is in the range of '0' to '9', 'A' to 'Z' or 'a' to 'z', or '_'
		return isDigitChar(c) || isValidNameStartChar(c);
	}
}