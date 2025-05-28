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
#include <slu/comp/ConvData.hpp>

namespace slu::comp::lua
{
	template<typename T>
	concept AnyIgnoredStatement = 
		std::same_as<T,parse::StatementType::GOTOv<true>>
		|| std::same_as<T,parse::StatementType::UNSAFE_LABEL>
		|| std::same_as<T,parse::StatementType::SAFE_LABEL>;

	struct ConvData : CommonConvData
	{
		parse::LuaMpDb& luaDb;
		parse::Output<>& out;
	};

	inline void convStat(const ConvData& conv)
	{
		ezmatch(conv.stat.data)(

		varcase(const auto&) {},
			//Ignore these
		varcase(const AnyIgnoredStatement auto&) {}
		);
	}
	inline void conv(const ConvData& conv)
	{
		//TODO: Implement the conversion logic here
	}
}