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

#include <slu/parser/State.hpp>
#include <slu/parser/Input.hpp>
#include <slu/parser/adv/SkipSpace.hpp>
#include <slu/parser/adv/ReadExprBase.hpp>
#include <slu/parser/adv/RequireToken.hpp>
#include <slu/parser/adv/ReadExpr.hpp>
#include <slu/parser/adv/ReadTable.hpp>
#include <slu/parser/adv/ReadTraitExpr.hpp>
#include <slu/parser/errors/CharErrors.h>
#include <slu/parser/errors/KwErrors.h>

namespace slu::parse
{
	//No space skip!
	//YOU parse the 'fn'
	template<bool isBasic,AnyInput In>
	inline ExprType::FnType readFnType(In& in, const OptSafety safety)
	{// [safety] "fn" typeExp "->" typeExp
		ExprType::FnType res{};
		res.safety = safety;

		res.argType = std::make_unique<parse::Expr>(readExpr(in, false));
		requireToken(in, "->");
		res.retType = std::make_unique<parse::Expr>(readExpr<isBasic>(in, false));

		return res;
	}
}