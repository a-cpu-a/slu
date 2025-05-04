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

#include <slua/parser/State.hpp>
#include <slua/parser/Input.hpp>
#include <slua/parser/adv/SkipSpace.hpp>
#include <slua/parser/adv/ReadExprBase.hpp>
#include <slua/parser/adv/RequireToken.hpp>
#include <slua/parser/adv/ReadTypeExpr.hpp>
#include <slua/parser/adv/ReadName.hpp>
#include <slua/parser/errors/CharErrors.h>

namespace slua::parse
{
	template<class T,bool NAMED, AnyInput In>
	inline T readFieldsDestr(In& in, auto&& ty)
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

				ret.items.emplace_back(fieldName, readPat(in));
			}
			else
				ret.items.emplace_back(readPat(in));

		} while (checkReadToken(in,","));

		requireToken(in, "}");
		skipSpace(in);

		if (isValidNameChar(in.peek()))
			ret.name = in.genData.resolveUnknown(readName(in));
		else
			ret.name = { SIZE_MAX };

		return ret;
	}
	template<bool TYPE_EXPR,AnyInput In>
	inline Pat readPatPastExpr(In& in,auto&& ty)
	{
		skipSpace(in);

		const char firstChar = in.peek();
		if (firstChar == '{')
		{
			in.skip();
			skipSpace(in);
			if (in.peek() == '|')
				return readFieldsDestr<DestrPatType::Fields,true>(in,std::move(ty));

			return readFieldsDestr<DestrPatType::List, false>(in, std::move(ty));
		}
		else if (firstChar == '}' || firstChar == ',')
		{
			if constexpr(!TYPE_EXPR)
				return PatType::Simple{ std::move(ty) };
			throwExpectedPatDestr(in);
		}
		//Must be Name then

		MpItmId<In> name = in.genData.resolveUnknown(readName(in));
		return PatType::DestrName{ name,std::move(ty)};
	}
	//Will not skip space!
	template<AnyInput In>
	inline Pat readPat(In& in)
	{
		const char firstChar = in.peek();

		if (firstChar == 'a' && checkReadTextToken("as"))
		{
			TypeExpr ty = readTypeExpr(in, true);

			return readPatPastExpr<true>(in, std::move(ty));
		}
		else if (firstChar == '_' && !isValidNameChar(in.peekAt(1)))
		{
			return PatType::DestrAny{};
		}

		Expression<In> expr = readExpr<true>(in, false);
		//TODO: type prefix form parsing

		return readPatPastExpr<false>(in,std::move(expr));
	}
}