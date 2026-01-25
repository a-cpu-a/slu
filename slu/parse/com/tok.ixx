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
#include <format>

#include <slu/Ansi.hpp>
export module slu.parse.com.tok;

import slu.ast.state;
import slu.parse.input;
import slu.parse.error;
import slu.parse.com.skip_space;
import slu.char_info;

namespace slu::parse
{
	export template<bool SKIP_SPACE = true, size_t TOK_SIZE>
	void requireToken(AnyInput auto& in, const char (&tok)[TOK_SIZE])
	{
		if constexpr (SKIP_SPACE)
			skipSpace(in);

		try
		{
			for (size_t i = 0; i < TOK_SIZE - 1; i++) //skip null
			{
				const char ch = in.get();
				if (ch != tok[i])
				{
					throw UnexpectedCharacterError(std::format(
					    "Expected " LUACC_SINGLE_STRING("{}") ", character " LUACC_SINGLE_STRING(
					        "{}") ", but found " LUACC_SINGLE_STRING("{}") "{}",
					    tok, tok[i], ch, errorLocStr(in)));
				}
			}
		} catch (EndOfStreamError&)
		{
			throw UnexpectedFileEndError(std::format(
			    "Expected " LUACC_SINGLE_STRING("{}") ", but file ended", tok));
		}
	}
	//Will NOT skip space!!!
	export [[nodiscard]] bool checkToken(AnyInput auto& in,
	    const std::string_view tok, const bool nameLike = false,
	    const bool readIfGood = false)
	{
		size_t off = 0;

		for (size_t i = 0; i < tok.size(); i++) //skip null
		{
			if (in.isOob(off))
				return false;

			if (in.peekAt(off++) != tok[i])
				return false;
		}


		if (nameLike)
		{
			if (!in.isOob(off))
			{

				const uint8_t ch = in.peekAt(off);
				if (isValidNameChar(ch))
					return false;
			}
		}

		if (readIfGood)
			in.skip(off);


		return true;
	}
	//Will NOT skip space!!!
	export template<size_t TOK_SIZE>
	[[nodiscard]] bool checkToken(AnyInput auto& in,
	    const char (&tok)[TOK_SIZE], const bool nameLike = false,
	    const bool readIfGood = false)
	{
		return checkToken(
		    in, std::string_view(tok, TOK_SIZE - 1), nameLike, readIfGood);
	}
	//Will NOT skip space!!!
	export template<size_t TOK_SIZE>
	[[nodiscard]] bool checkTextToken(
	    AnyInput auto& in, const char (&tok)[TOK_SIZE])
	{
		return checkToken(in, tok, true);
	}
	//Will NOT skip space!!!
	export [[nodiscard]] bool checkTextToken(
	    AnyInput auto& in, const std::string_view tok)
	{
		return checkToken(in, tok, true);
	}

	export template<bool SKIP_SPACE = true, size_t TOK_SIZE>
	[[nodiscard]] bool checkReadToken(AnyInput auto& in,
	    const char (&tok)[TOK_SIZE], const bool nameLike = false)
	{
		if constexpr (SKIP_SPACE)
			skipSpace(in);
		return checkToken(in, tok, nameLike, true);
	}
	export template<bool SKIP_SPACE = true, size_t TOK_SIZE>
	[[nodiscard]] bool checkReadTextToken(
	    AnyInput auto& in, const char (&tok)[TOK_SIZE])
	{
		if constexpr (SKIP_SPACE)
			skipSpace(in);
		return checkToken(in, tok, true, true);
	}

	export template<bool SKIP_SPACE = true, size_t TOK_SIZE>
	void readOptToken(AnyInput auto& in, const char (&tok)[TOK_SIZE],
	    const bool nameLike = false)
	{
		(void)checkReadToken<SKIP_SPACE>(in, tok, nameLike);
	}
} //namespace slu::parse