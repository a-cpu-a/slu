module;
/*
** See Copyright Notice inside Include.hpp
*/
#include <cstdint>
#include <unordered_set>
#include <memory>

#include <slu/parse/adv/ReadExpr.hpp>
#include <slu/parse/adv/ReadTable.hpp>
export module slu.parse.basic.fn_ty;

import slu.ast.state;
import slu.parse.input;
import slu.parse.adv.expr_base;
import slu.parse.com.skip_space;
import slu.parse.com.tok;

namespace slu::parse
{
	//No space skip!
	//YOU parse the 'fn'
	export template<bool isBasic,AnyInput In>
	ExprType::FnType readFnType(In& in, const ast::OptSafety safety)
	{// [safety] "fn" typeExp "->" typeExp
		ExprType::FnType res{};
		res.safety = safety;

		res.argType = std::make_unique<parse::Expr>(readExpr(in, false));
		requireToken(in, "->");
		res.retType = std::make_unique<parse::Expr>(readExpr<isBasic>(in, false));

		return res;
	}
}