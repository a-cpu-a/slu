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
#include <slu/parse/adv/RequireToken.hpp>

namespace slu::parse
{
	template<AnyInput In>
	inline TraitExpr readTraitExpr(In& in)
	{
		skipSpace(in);
		TraitExpr ret;
		ret.place = in.getLoc();

		ret.traitCombo.emplace_back(readBasicExpr(in, false, false));

		while (in)
		{
			skipSpace(in);
			if (in.peek() != '+')
				break;
			in.skip();

			ret.traitCombo.emplace_back(readBasicExpr(in, false, false));
		}
		return ret;
	}
}