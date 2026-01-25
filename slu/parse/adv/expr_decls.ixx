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
export module slu.parse.adv.expr_decls;

import slu.ast.state_decls;
import slu.parse.input;

namespace slu::parse
{
	extern "C++"
	{
	export template<bool IS_BASIC = false, bool FOR_PAT = false, AnyInput In>
	parse::Expr readExpr(
	    In& in, const bool allowVarArg, const bool readBiOp = true);
	}
} //namespace slu::parse