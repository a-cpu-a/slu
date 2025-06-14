﻿/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <slu/lang/BasicState.hpp>
#include <slu/parser/State.hpp>
#include <slu/parser/BasicGenData.hpp>

#include <slu/comp/CompCfg.hpp>

namespace slu::comp
{
	struct CommonConvData
	{
		const CompCfg& cfg;
		const parse::BasicMpDbData& sharedDb;
		const parse::StatementV<true>& stat;
	};
}