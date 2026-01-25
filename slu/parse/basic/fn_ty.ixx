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
#include <memory>
#include <unordered_set>

export module slu.parse.basic.fn_ty;

import slu.ast.enums;
import slu.ast.state;
import slu.parse.input;
import slu.parse.adv.expr_decls;
import slu.parse.com.skip_space;
import slu.parse.com.tok;

namespace slu::parse
{
	//No space skip!
	//YOU parse the 'fn'
	export template<bool isBasic, AnyInput In>
	ExprType::FnType readFnType(In& in, const ast::OptSafety safety)
	{ // [safety] "fn" typeExp "->" typeExp
		ExprType::FnType res{};
		res.safety = safety;

		res.argType = std::make_unique<parse::Expr>(readExpr(in, false));
		requireToken(in, "->");
		res.retType
		    = std::make_unique<parse::Expr>(readExpr<isBasic>(in, false));

		return res;
	}
} //namespace slu::parse