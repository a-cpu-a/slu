module;
/*
** See Copyright Notice inside Include.hpp
*/
#include <string_view>

export module slu.comp.conv_data;

import slu.ast.mp_data;
import slu.ast.state;
import slu.comp.cfg;

namespace slu::comp
{
	export struct CommonConvData
	{
		const CompCfg& cfg;
		const parse::BasicMpDbData& sharedDb;
		const parse::Stat* stat;
		std::string_view filePath;
	};
}