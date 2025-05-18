/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <cstdint>
#include <unordered_set>

//https://www.lua.org/manual/5.4/manual.html
//https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form
//https://www.sciencedirect.com/topics/computer-science/backus-naur-form

#include <slua/parser/State.hpp>
#include <slua/parser/Input.hpp>
#include <slua/parser/adv/SkipSpace.hpp>
#include <slua/parser/adv/RequireToken.hpp>
#include <slua/parser/adv/ReadName.hpp>

namespace slua::parse
{
	template<AnyInput In>
	inline void readLabel(In& in, const Position place)
	{
		//label ::= ‘::’ Name ‘::’
		//SL label ::= ‘:::’ Name ‘:’

		requireToken(in, sel<In>("::", ":::"));

		if constexpr (in.settings() & sluaSyn)
		{
			if (checkReadTextToken(in, "unsafe"))
			{
				return in.genData.addStat(place, StatementType::UNSAFE_LABEL{});
			}
			else if (checkReadTextToken(in, "safe"))
			{
				return in.genData.addStat(place, StatementType::SAFE_LABEL{});
			}
		}

		const MpItmId<In> res = in.genData.resolveUnknown(readName(in));

		requireToken(in, sel<In>("::", ":"));

		return in.genData.addStat(place, StatementType::LABEL<In>{res});
	}
}