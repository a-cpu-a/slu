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
#include <unordered_set>

#include <slu/Ansi.hpp>
export module slu.parse.adv.tbl;

import slu.char_info;
import slu.ast.state;
import slu.ast.state_decls;
import slu.lang.basic_state;
import slu.parse.error;
import slu.parse.input;
import slu.parse.adv.expr_decls;
import slu.parse.com.skip_space;
import slu.parse.com.tok;

namespace slu::parse
{
	export template<AnyInput In>
	Field<In> readField(In& in, const bool allowVarArg)
	{
		// field :: = ‘[’ exp ‘]’ ‘ = ’ exp | Name ‘ = ’ exp | exp
		skipSpace(in);

		/*if (checkReadToken(in, "["))
		{
		    FieldType::Expr2Expr res{};

		    res.idx = readExpr(in, allowVarArg);

		    requireToken(in, "]");
		    requireToken(in, "=");

		    res.v = readExpr(in, allowVarArg);

		    return res;
		}*/

		const size_t nameOffset = peekName(in);

		if (nameOffset != SIZE_MAX)
		{
			//Lazy-TODO: eof handling lol
			const size_t spacedOffset = weakSkipSpace(in, nameOffset);

			//check at the CORRECT position, AND that it ISNT ==
			if (in.peekAt(spacedOffset) == '='
			    && in.peekAt(spacedOffset + 1) != '=')
			{
				const lang::MpItmId name
				    = in.genData.resolveUnknown(readName(in));
				skipSpace(in);
				in.skip(); // '='

				return FieldType::Name2Expr(name, readExpr(in, allowVarArg));
			}
		}
		return FieldType::Expr(readExpr(in, allowVarArg));
	}

	//Will NOT check the first char '{' !!!
	//But will skip it
	export template<bool skipStart = true, AnyInput In>
	Table<In> readTable(In& in, const bool allowVarArg)
	{
		if (skipStart)
			in.skip(); //get rid of '{'

		skipSpace(in);

		Table<In> tbl{};

		if (in.peek() == '}')
		{
			in.skip();
			return tbl;
		}
		//must be field
		tbl.emplace_back(readField(in, allowVarArg));

		while (true)
		{
			skipSpace(in);
			const char ch = in.peek();
			if (ch == '}')
			{
				in.skip();
				break;
			}
			if (!isFieldSep(ch))
			{
				throw UnexpectedCharacterError(std::format(
				    "Expected table separator (" LUACC_SINGLE_STRING(",") " or " LUACC_SINGLE_STRING(
				        ";") "), found " LUACC_SINGLE_STRING("{}") "{}",
				    ch, errorLocStr(in)));
			}
			in.skip(); //skip field-sep

			skipSpace(in);
			if (in.peek() == '}')
			{
				in.skip();
				break;
			}
			tbl.emplace_back(readField(in, allowVarArg));
		}
		return tbl;
	}
} //namespace slu::parse