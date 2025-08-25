/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <cstdint>
#include <unordered_set>

//https://www.lua.org/manual/5.4/manual.html
//https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form
//https://www.sciencedirect.com/topics/computer-science/backus-naur-form

#include <slu/parse/State.hpp>
import slu.parse.input;
#include <slu/parse/adv/SkipSpace.hpp>
#include <slu/parse/adv/RequireToken.hpp>
#include <slu/parse/adv/ReadName.hpp>

namespace slu::parse
{
	template<AnyInput In>
	inline void readLabel(In& in, const ast::Position place)
	{
		//label ::= ‘::’ Name ‘::’
		//SL label ::= ‘:::’ Name ‘:’

		requireToken(in, ":::");

		if (checkReadTextToken(in, "unsafe"))
		{
			requireToken(in, ":");
			in.genData.setUnsafe();
			return in.genData.addStat(place, StatType::UnsafeLabel{});
		}
		else if (checkReadTextToken(in, "safe"))
		{
			requireToken(in, ":");
			in.genData.setSafe();
			return in.genData.addStat(place, StatType::SafeLabel{});
		}

		const MpItmId res = in.genData.addLocalObj(readName(in));

		requireToken(in, ":");

		return in.genData.addStat(place, StatType::Label<In>{res});
	}
}