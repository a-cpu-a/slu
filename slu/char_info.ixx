module;
/*
** See Copyright Notice inside Include.hpp
*/
#include <cstdint>

export module slu.char_info;

namespace slu
{
	export constexpr bool isSpaceChar(const char ch) {
		return ch == ' ' || ch == '\f' || ch == '\t' || ch == '\v' || ch == '\n' || ch == '\r';
	}

	constexpr uint8_t CAPITAL_BIT = 'x' - 'X';
	static_assert('x' - 'X' == 32);//is simple bit flip?

	export constexpr char numToHex(const uint8_t v) {
		if (v <= 9)
			return ('0' + v);
		return ('A' + (v - 10));
	}
	export constexpr char toLowerCase(const char c) {
		return c | CAPITAL_BIT;
	}

	export constexpr char toUpperCase(const char c) {
		return c & (~CAPITAL_BIT);
	}

	export constexpr bool isLowerCaseEqual(const char charToCheck,const char c) {
		return toLowerCase(charToCheck)==c;
	}
	export constexpr bool isDigitChar(const char c) {
		return c >= '0' && c <= '9';
	}
	export constexpr bool isAlpha(const char c) {
		return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
	}
	export constexpr bool isHexDigitChar(const char c) {
		return isDigitChar(c) || (c >= 'A' && c <= 'F') || (c >= 'a' && c <= 'f');
	}

	static_assert('9' < 'a');
	static_assert('9' < 'A');
	export constexpr uint8_t hexDigit2Num(const char c)
	{
		if (c <= '9')
			return c - '0';//to num
		return toLowerCase(c) -'a' +10;
	}
	export constexpr bool isValidNameStartChar(const char c)
	{// Check if the character is in the range of 'A'..'Z' or 'a'..'z', or '_'
		return isAlpha(c) || c == '_';
	}
	export constexpr bool isValidNameChar(const char c)
	{// Check if the character is in the range of '0'..'9', 'A'..'Z' or 'a'..'z', or '_'
		return isDigitChar(c) || isValidNameStartChar(c);
	}
}