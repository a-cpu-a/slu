/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <cstdint>
#include <array>
#include <string_view>
#include <slu/parser/Enums.hpp>
#include <slu/parser/basic/CharInfo.hpp>

namespace slu::parse
{
	// All the names with '_' or marked with !!! are special, so these lists are not exactly for them
	// Note: due to how ascii works _ turns to ?
	constexpr void handleTraitCase(auto& res, bool forTrait)
	{
		if (forTrait)
		{
			for (auto& i : res)
				i[0] = parse::toUpperCase(i[0]);
		}
	}
	constexpr auto mkUnOpNames(bool forTrait)
	{
		std::array<std::array<char,10>, (size_t)UnOpType::ENUM_SIZE> res = {
			"neg","not", "_len", "_bitNot",

			"rangeMax",//!!!

			"_alloc",
			"ref", "refMut", "refConst","refShare",
			"ptr", "ptrMut", "ptrConst","ptrShare"
		};
		handleTraitCase(res,forTrait);
		return res;
	}
	constexpr auto mkPostUnOpNames(bool forTrait)
	{
		std::array<std::array<char,10>, (size_t)PostUnOpType::ENUM_SIZE> res = {
			"rangeMin",//!!!
			"_deref", "_try"
		};
		handleTraitCase(res,forTrait);
		return res;
	}
	constexpr auto mkPostBinOpNames(bool forTrait)
	{
		std::array<std::array<char,14>, (size_t)BinOpType::ENUM_SIZE> res = {
			"add","sub", "mul", "div","flrDiv", "pow", "rem",
			"bitAnd", "bitXor", "bitOr", "shr", "shl",
			"concat",

			"lt", "le", "gt", "ge", "eq", "ne",//!!!
			"and","or",//!!!
			
			"rep",
			"range",//!!!
			"_mkResult"
		};
		handleTraitCase(res,forTrait);
		return res;
	}

	constexpr auto unOpNames = mkUnOpNames(false);
	constexpr auto unOpTraitNames = mkUnOpNames(true);
	
	constexpr auto postUnOpNames = mkPostUnOpNames(false);
	constexpr auto postUnOpTraitNames = mkPostUnOpNames(true);

	constexpr auto binOpNames = mkPostBinOpNames(false);
	constexpr auto binOpTraitNames = mkPostBinOpNames(true);
}