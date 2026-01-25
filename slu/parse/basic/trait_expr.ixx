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
#include <memory>
#include <unordered_set>

export module slu.parse.basic.trait_expr;

import slu.ast.state;
import slu.parse.input;
import slu.parse.adv.expr_base;
import slu.parse.com.skip_space;
import slu.parse.com.tok;

namespace slu::parse
{
	export template<AnyInput In> TraitExpr readTraitExpr(In& in)
	{
		skipSpace(in);
		TraitExpr ret;
		ret.place = in.getLoc();

		ret.traitCombo.emplace_back(readBasicExpr(in, false, false));

		while (in)
		{
			skipSpace(in);
			if (in.peek() != '+')
				break;
			in.skip();

			ret.traitCombo.emplace_back(readBasicExpr(in, false, false));
		}
		return ret;
	}
} //namespace slu::parse