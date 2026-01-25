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
#include <utility>

export module slu.parse.basic.struct_stat;

import slu.ast.pos;
import slu.ast.state;
import slu.lang.basic_state;
import slu.parse.input;
import slu.parse.com.name;
import slu.parse.com.skip_space;
import slu.parse.com.tok;

namespace slu::parse
{

	//Doesnt check (, starts after it too.
	export template<bool isLocal, AnyInput In> ParamList readParamList(In& in)
	{
		ParamList res;
		skipSpace(in);

		if (in.peek() == ')')
		{
			in.skip();
			return res;
		}

		res.emplace_back(readFuncParam(in));

		while (checkReadToken(in, ","))
			res.emplace_back(readFuncParam(in));

		requireToken(in, ")");

		return res;
	}
	export template<class T, bool structOnly, AnyInput In>
	void readStructStat(
	    In& in, const ast::Position place, const lang::ExportData exported)
	{
		in.genData.pushLocalScope();
		T res{};
		res.exported = exported;

		res.name = in.genData.addLocalObj(readName(in));

		skipSpace(in);
		if (in.peek() == '(')
		{
			in.skip();

			res.params = readParamList<true>(in);
			skipSpace(in);
		}
		requireToken(in, "{");
		res.type = readTable<false>(in, false);
		res.local2Mp = in.genData.popLocalScope();

		in.genData.addStat(place, std::move(res));
	}
} //namespace slu::parse