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

import slu.char_info;
import slu.ast.state;
import slu.parse.input;
import slu.parse.adv.expr_base;
import slu.parse.com.name;
import slu.parse.com.skip_space;
import slu.parse.com.tok;

namespace slu::parse
{
	template<bool isLocal, class T,bool NAMED, AnyInput In>
	inline T readFieldsDestr(In& in, auto&& ty, const bool uncond)
		requires(AnyCompoundDestr<isLocal,T>)
	{
		T ret;
		ret.spec = std::move(ty);
		do
		{
			skipSpace(in);
			if (in.peekAt(2) != '.' && checkReadToken(in,".."))
			{
				ret.extraFields = true;
				break;
			}
			if constexpr(NAMED)
			{
				requireToken(in, "|");
				skipSpace(in);
				lang::PoolString fieldName = in.genData.poolStr(readSluTuplableName(in));
				requireToken(in, "|");
				skipSpace(in);

				ret.items.emplace_back(fieldName, readPat<isLocal>(in, uncond));
			}
			else
				ret.items.emplace_back(readPat<isLocal>(in, uncond));

		} while (checkReadToken(in,","));

		requireToken(in, "}");
		skipSpace(in);

		if (isValidNameChar(in.peek()) && !checkTextToken(in,"in"))
			ret.name = in.genData.template resolveNewName<isLocal>(readName(in));
		else
			ret.name = in.genData.template resolveNewSynName<isLocal>();

		return ret;
	}
	template<bool isLocal,bool IS_EXPR,AnyInput In>
	inline Pat<In, isLocal> readPatPastExpr(In& in,auto&& ty,const bool uncond)
	{
		skipSpace(in);

		const char firstChar = in.peek();
		if (firstChar == '{')
		{
			in.skip();
			skipSpace(in);
			if (in.peek() == '|')
				return readFieldsDestr<isLocal,DestrPatType::Fields<In, isLocal>,true>(in,std::move(ty), uncond);

			return readFieldsDestr<isLocal,DestrPatType::List<In, isLocal>, false>(in, std::move(ty), uncond);
		}
		else if (firstChar == ')' || firstChar == '}' || firstChar == ',')
		{
			if constexpr(IS_EXPR)
			{
				if (!uncond)
					return PatType::Simple<In>{ std::move(ty) };
			}
			throwExpectedPatDestr(in);
		}
		//Must be Name then

		auto nameOrLocal = in.genData.template resolveNewName<isLocal>(readName(in));

		if(!uncond)
		{
			skipSpace(in);
			if (in.peek() == '=')
				return PatType::DestrNameRestrict<In, isLocal>{ {nameOrLocal,std::move(ty)},readExpr(in,false) };
		}

		return PatType::DestrName<In, isLocal>{ nameOrLocal,std::move(ty) };
	}
	//Will not skip space!
	template<bool isLocal, AnyInput In>
	inline Pat<In,isLocal> readPat(In& in, const bool uncond)
	{
		const char firstChar = in.peek();

		if (firstChar == '_' && !isValidNameChar(in.peekAt(1)))
		{
			in.skip();
			return PatType::DestrAny<In, isLocal>{in.genData.template resolveNewSynName<isLocal>()};
		}

		Expr expr = readExpr<true,true>(in, false);

		if (std::holds_alternative<ExprType::PatTypePrefix>(expr.data))
		{
			return readPatPastExpr<isLocal,false>(in, TypePrefix(std::move(expr.unOps)), uncond);
		}

		return readPatPastExpr<isLocal,true>(in,std::move(expr), uncond);
	}
}