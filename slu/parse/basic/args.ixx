/*
    Slu language compiler, a computer program compiler.
    Copyright (C) 2026 a-cpu-a <any1word@proton.me>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

      SPDX-License-Identifier: AGPL3.0-or-later
*/
module;
#include <cstdint>
#include <format>

#include <slu/Ansi.hpp>
export module slu.parse.basic.args;

import slu.ast.state;
import slu.parse.error;
import slu.parse.input;
import slu.parse.com.name;
import slu.parse.com.skip_space;
import slu.parse.com.tok;

namespace slu::parse
{
	//Doesnt skip space, the current character must be a valid args starter
	export template<AnyInput In>
	inline Args readArgs(In& in, const bool allowVarArg)
	{
		const char ch = in.peek();
		if (ch == '"' || ch == '\'' || ch == '[')
		{
			return ArgsType::String(readStringLiteral(in, ch), in.getLoc());
		} else if (ch == '(')
		{
			in.skip(); //skip start
			skipSpace(in);
			ArgsType::ExprList res{};
			if (in.peek() == ')') // Check if 0 args
			{
				in.skip();
				return res;
			}
			res = readExprList(in, allowVarArg);
			requireToken(in, ")");
			return res;
		} else if (ch == '{')
		{
			return ArgsType::Table<In>(readTable(in, allowVarArg));
		}
		throw UnexpectedCharacterError(std::format(
		    "Expected function arguments (" LUACC_SINGLE_STRING(",") " or " LUACC_SINGLE_STRING(
		        ";") "), found " LUACC_SINGLE_STRING("{}") "{}",
		    ch, errorLocStr(in)));
	}
} //namespace slu::parse