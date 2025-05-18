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
#include <slu/parser/adv/ReadTypeExpr.hpp>
#include <slu/parser/adv/ReadName.hpp>
#include <slu/parser/errors/CharErrors.h>

namespace slua::parse
{
	template<AnyCompoundDestr T,bool NAMED, AnyInput In>
	inline T readFieldsDestr(In& in, auto&& ty, const bool uncond)
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
				MpItmId<In> fieldName = in.genData.resolveUnknown(readName(in));
				requireToken(in, "|");
				skipSpace(in);

				ret.items.emplace_back(fieldName, readPat(in, uncond));
			}
			else
				ret.items.emplace_back(readPat(in, uncond));

		} while (checkReadToken(in,","));

		requireToken(in, "}");
		skipSpace(in);

		if (isValidNameChar(in.peek()))
			ret.name = in.genData.resolveUnknown(readName(in));
		else
			ret.name = { SIZE_MAX };

		return ret;
	}
	template<bool IS_EXPR,AnyInput In>
	inline Pat readPatPastExpr(In& in,auto&& ty,const bool uncond)
	{
		skipSpace(in);

		const char firstChar = in.peek();
		if (firstChar == '{')
		{
			in.skip();
			skipSpace(in);
			if (in.peek() == '|')
				return readFieldsDestr<DestrPatType::Fields,true>(in,std::move(ty), uncond);

			return readFieldsDestr<DestrPatType::List, false>(in, std::move(ty), uncond);
		}
		else if (firstChar == ')' || firstChar == '}' || firstChar == ',')
		{
			if constexpr(IS_EXPR)
			{
				if (!uncond)
					return PatType::Simple{ std::move(ty) };
			}
			throwExpectedPatDestr(in);
		}
		//Must be Name then

		MpItmId<In> name = in.genData.resolveUnknown(readName(in));

		if(!uncond)
		{
			skipSpace(in);
			if (in.peek() == '=')
				return PatType::DestrNameRestrict{ {name,std::move(ty)},readExpr(in,false) };
		}

		return PatType::DestrName{ name,std::move(ty) };
	}
	//Will not skip space!
	template<AnyInput In>
	inline Pat readPat(In& in, const bool uncond)
	{
		const char firstChar = in.peek();

		if (firstChar == 'a' && checkReadTextToken(in,"as"))
		{
			TypeExpr ty = readTypeExpr(in, true);

			return readPatPastExpr<false>(in, std::move(ty), uncond);
		}
		else if (firstChar == '_' && !isValidNameChar(in.peekAt(1)))
		{
			return PatType::DestrAny{};
		}

		Expression<In> expr = readExpr<true,true>(in, false);

		if (std::holds_alternative<ExprType::PAT_TYPE_PREFIX>(expr.data))
		{
			return readPatPastExpr<false>(in, TypePrefix(std::move(expr.unOps)), uncond);
		}

		return readPatPastExpr<true>(in,std::move(expr), uncond);
	}
}