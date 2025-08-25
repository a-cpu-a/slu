/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <slu/parse/State.hpp>
#include <slu/parse/BasicGenData.hpp>

#include <slu/comp/CompCfg.hpp>

namespace slu::comp
{
	struct CommonConvData
	{
		const CompCfg& cfg;
		const parse::BasicMpDbData& sharedDb;
		const parse::Stat* stat;
		std::string_view filePath;
	};
}