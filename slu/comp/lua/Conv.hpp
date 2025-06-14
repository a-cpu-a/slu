﻿/*
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
		|| std::same_as<T,parse::StatementType::SEMICOLON>
		|| std::same_as<T,parse::StatementType::USE>
		|| std::same_as<T,parse::StatementType::FnDeclV<true>>
		|| std::same_as<T,parse::StatementType::FunctionDeclV<true>>
		|| std::same_as<T,parse::StatementType::ExternBlockV<true>>//ignore, as desugaring will remove it
		|| std::same_as<T,parse::StatementType::UnsafeBlockV<true>>//ignore, as desugaring will remove it
		|| std::same_as<T,parse::StatementType::DROPv<true>>
		|| std::same_as<T,parse::StatementType::MOD_DEFv<true>>
		|| std::same_as<T,parse::StatementType::MOD_DEF_INLINEv<true>>
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