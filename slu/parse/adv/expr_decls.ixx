module;
/*
** See Copyright Notice inside Include.hpp
*/
export module slu.parse.adv.expr_decls;

import slu.ast.state_decls;
import slu.parse.input;

namespace slu::parse
{
	extern "C++" {
		export template<bool IS_BASIC = false, bool FOR_PAT = false, AnyInput In>
			parse::Expr readExpr(In& in, const bool allowVarArg, const bool readBiOp = true);
	}
}