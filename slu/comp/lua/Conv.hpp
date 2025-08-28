/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <string>
#include <vector>
#include <optional>
#include <thread>
#include <variant>

import slu.comp.cfg;
import slu.comp.conv_data;

namespace slu::comp::lua
{
	template<typename T>
	concept AnyIgnoredStat = 
		std::same_as<T,parse::StatType::GotoV<true>>
		|| std::same_as<T,parse::StatType::Semicol>
		|| std::same_as<T,parse::StatType::Use>
		|| std::same_as<T,parse::StatType::FnDeclV<true>>
		|| std::same_as<T,parse::StatType::FunctionDeclV<true>>
		|| std::same_as<T,parse::StatType::ExternBlockV<true>>//ignore, as desugaring will remove it
		|| std::same_as<T,parse::StatType::UnsafeBlockV<true>>//ignore, as desugaring will remove it
		|| std::same_as<T,parse::StatType::DropV<true>>
		|| std::same_as<T,parse::StatType::ModV<true>>
		|| std::same_as<T,parse::StatType::ModAsV<true>>
		|| std::same_as<T,parse::StatType::UnsafeLabel>
		|| std::same_as<T,parse::StatType::SafeLabel>;

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
		varcase(const AnyIgnoredStat auto&) {}
		);
	}
	inline void conv(const ConvData& conv)
	{
		//TODO: Implement the conversion logic here
	}
}