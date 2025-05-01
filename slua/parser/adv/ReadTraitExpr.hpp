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
#include <slua/parser/errors/CharErrors.h>

namespace slua::parse
{
	template<AnyInput In>
	inline TraitExpr readTraitExpr(In& in)
	{
		const Position startPos = in.getLoc();
		
		TraitExpr ret;
		ret.place = startPos;

		const char firstChar = in.peek();
		
		if (firstChar != '(' && !isValidNameStartChar(firstChar))
			throw throwExpectedTraitExpr(in);

		ret.traitCombo.emplace_back(parsePrefixExprVar<TraitExprItem, true,true>(in, false, firstChar));

		while (true)
		{
			skipSpace(in);

			if (!in)
				break;//File ended

			if (in.peek() != '+')
				break;
			in.skip();

			ret.traitCombo.emplace_back(parsePrefixExprVar<TraitExprItem, true, true>(in, false, firstChar));

		}

		return ret;
	}
}