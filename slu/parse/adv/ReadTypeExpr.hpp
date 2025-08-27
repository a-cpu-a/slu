/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <cstdint>
#include <unordered_set>
#include <memory>

//https://www.lua.org/manual/5.4/manual.html
//https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form
//https://www.sciencedirect.com/topics/computer-science/backus-naur-form

import slu.ast.state;
import slu.parse.input;
import slu.parse.com.skip_space;
#include <slu/parse/adv/ReadExprBase.hpp>
import slu.parse.com.token;
#include <slu/parse/adv/ReadExpr.hpp>
#include <slu/parse/adv/ReadTable.hpp>
#include <slu/parse/adv/ReadTraitExpr.hpp>

namespace slu::parse
{
	//No space skip!
	//YOU parse the 'fn'
	template<bool isBasic,AnyInput In>
	inline ExprType::FnType readFnType(In& in, const ast::OptSafety safety)
	{// [safety] "fn" typeExp "->" typeExp
		ExprType::FnType res{};
		res.safety = safety;

		res.argType = std::make_unique<parse::Expr>(readExpr(in, false));
		requireToken(in, "->");
		res.retType = std::make_unique<parse::Expr>(readExpr<isBasic>(in, false));

		return res;
	}
}