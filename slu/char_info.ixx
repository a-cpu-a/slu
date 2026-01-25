/*
    A program file.
    Copyright (C) 2026 a-cpu-a <any1word@proton.me>

    This file is part of Slu-c.

    Slu-c is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Slu-c is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with Slu-c.  If not, see <https://www.gnu.org/licenses/>.

      SPDX-License-Identifier: AGPL3.0-or-later
*/
module;
#include <cstdint>

export module slu.char_info;

namespace slu
{
	export constexpr bool isFieldSep(const char ch)
	{
		return ch == ',' || ch == ';';
	}
	export constexpr bool isStrStarter(const char ch1)
	{
		return ch1 == '=' || ch1 == '[' || ch1 == '\'' || ch1 == '"';
	}
	export constexpr bool isSpaceChar(const char ch)
	{
		return ch == ' ' || ch == '\f' || ch == '\t' || ch == '\v' || ch == '\n'
		    || ch == '\r';
	}

	constexpr uint8_t CAPITAL_BIT = 'x' ^ 'X';
	static_assert(CAPITAL_BIT == 32); // is classic ascii?

	export constexpr char numToHex(const uint8_t v)
	{
		if (v <= 9)
			return ('0' + v);
		return ('A' + (v - 10));
	}
	export constexpr char toLowerCase(const char c)
	{
		return c | CAPITAL_BIT;
	}

	export constexpr char toUpperCase(const char c)
	{
		return c & (~CAPITAL_BIT);
	}

	export constexpr bool isLowerCaseEqual(const char charToCheck, const char c)
	{
		return toLowerCase(charToCheck) == c;
	}
	export constexpr bool isDigitChar(const char c)
	{
		return c >= '0' && c <= '9';
	}
	export constexpr bool isAlpha(const char c)
	{
		return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
	}
	export constexpr bool isHexDigitChar(const char c)
	{
		return isDigitChar(c) || (c >= 'A' && c <= 'F')
		    || (c >= 'a' && c <= 'f');
	}

	static_assert('9' < 'a');
	static_assert('9' < 'A');
	export constexpr uint8_t hexDigit2Num(const char c)
	{
		if (c <= '9')
			return c - '0'; //to num
		return toLowerCase(c) - 'a' + 10;
	}
	export constexpr bool isValidNameStartChar(const char c)
	{ // Check if the character is in the range of 'A'..'Z' or 'a'..'z', or '_'
		return isAlpha(c) || c == '_';
	}
	export constexpr bool isValidNameChar(const char c)
	{ // Check if the character is in the range
	  // of '0'..'9', 'A'..'Z' or 'a'..'z', or '_'
		return isDigitChar(c) || isValidNameStartChar(c);
	}
} //namespace slu