﻿/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <cstdint>
#include <unordered_set>

//https://www.lua.org/manual/5.4/manual.html
//https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form
//https://www.sciencedirect.com/topics/computer-science/backus-naur-form

#include <slu/parser/State.hpp>
#include <slu/parser/Input.hpp>
#include <slu/parser/adv/SkipSpace.hpp>
#include <slu/parser/adv/RequireToken.hpp>
#include <slu/parser/adv/ReadName.hpp>

namespace slu::parse
{
	template<AnyInput In>
	inline void readLabel(In& in, const Position place)
	{
		//label ::= ‘::’ Name ‘::’
		//SL label ::= ‘:::’ Name ‘:’

		requireToken(in, sel<In>("::", ":::"));

		if constexpr (In::settings() & sluSyn)
		{
			if (checkReadTextToken(in, "unsafe"))
			{
				requireToken(in, ":");
				in.genData.setUnsafe();
				return in.genData.addStat(place, StatementType::UNSAFE_LABEL{});
			}
			else if (checkReadTextToken(in, "safe"))
			{
				requireToken(in, ":");
				in.genData.setSafe();
				return in.genData.addStat(place, StatementType::SAFE_LABEL{});
			}
		}

		const MpItmId<In> res = in.genData.addLocalObj(readName(in));

		requireToken(in, sel<In>("::", ":"));

		return in.genData.addStat(place, StatementType::LABEL<In>{res});
	}
}