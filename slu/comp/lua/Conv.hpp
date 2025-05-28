/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <string>
#include <vector>
#include <optional>
#include <thread>
#include <variant>
#include <slu/lang/BasicState.hpp>

#include <slu/comp/CompCfg.hpp>

namespace slu::comp::lua
{
	inline void conv(const CompCfg& cfg,
		const parse::BasicMpDbData& sharedDb,
		parse::Output<>& out,
		const parse::StatementV<true>& stat)
	{
		//TODO: Implement the conversion logic here
	}
}