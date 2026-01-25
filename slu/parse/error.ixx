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
#include <string>

export module slu.parse.error;

namespace slu::parse
{
	export struct ParseFailError : std::exception
	{
		const char* what() const noexcept override
		{
			return "Failed to parse some slu/lua code.";
		}
	};

	export struct BasicParseError : std::exception
	{
		std::string m;
		BasicParseError(const std::string& m) : m(m) {}
		const char* what() const noexcept override
		{
			return m.c_str();
		}
	};
	export struct FailedRecoveryError : BasicParseError
	{
		using BasicParseError::BasicParseError;
	};

	export struct ParseError : BasicParseError
	{
		using BasicParseError::BasicParseError;
	};

#define MAKE_ERROR(_NAME)             \
	export struct _NAME : ParseError  \
	{                                 \
		using ParseError::ParseError; \
	}

	MAKE_ERROR(UnicodeError);
	MAKE_ERROR(UnexpectedKeywordError);
	MAKE_ERROR(UnexpectedCharacterError);
	MAKE_ERROR(UnexpectedFileEndError);
	MAKE_ERROR(ReservedNameError);
	MAKE_ERROR(ErrorWhileContext);
	MAKE_ERROR(InternalError);
} //namespace slu::parse