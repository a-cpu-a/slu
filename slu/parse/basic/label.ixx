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
#include <unordered_set>

export module slu.parse.basic.label;

import slu.ast.pos;
import slu.ast.state;
import slu.lang.basic_state;
import slu.parse.input;
import slu.parse.com.name;
import slu.parse.com.skip_space;
import slu.parse.com.tok;

namespace slu::parse
{
	export template<AnyInput In>
	void readLabel(In& in, const ast::Position place)
	{
		//label ::= ‘::’ Name ‘::’
		//SL label ::= ‘:::’ Name ‘:’

		requireToken(in, ":::");

		if (checkReadTextToken(in, "unsafe"))
		{
			requireToken(in, ":");
			in.genData.setUnsafe();
			return in.genData.addStat(place, StatType::UnsafeLabel{});
		} else if (checkReadTextToken(in, "safe"))
		{
			requireToken(in, ":");
			in.genData.setSafe();
			return in.genData.addStat(place, StatType::SafeLabel{});
		}

		const lang::MpItmId res = in.genData.addLocalObj(readName(in));

		requireToken(in, ":");

		return in.genData.addStat(place, parse::StatType::Label<In>{res});
	}
} //namespace slu::parse