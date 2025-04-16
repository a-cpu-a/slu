/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <string>
#include <unordered_map>
#include <variant>

#include <slua/parser/Parse.hpp>
#include <slua/midlevel/GenState.hpp>

namespace slua::mlvl
{
	inline void processStat(GenState& state,const parse::StatementV<true>& data)
	{
		ezmatch(data.data)(
		);
	}
	inline void processBlock(GenState& state,const parse::BlockV<true>& data)
	{
		for (const parse::StatementV<true>& stat : data.statList)
		{
			processStat(state, stat);
		}
		if (data.hadReturn)
		{
			//TODO
		}
	}
	inline void processFile(GenState& state,const parse::ParsedFileV<true>& data,const ModPath& thisMod)
	{
		ModPathId id = state.getMpIdLock(thisMod);
		processBlock(state, data.code);
	}
}