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
	constexpr auto nameArraysToSvs(auto& arrays)
	{
		std::array<std::string_view, sizeof(arrays)/sizeof(arrays[0])> res;
		for (size_t i = 0; i < arrays.size(); ++i)
		{
			res[i] = std::string_view(arrays[i].data());
		}
		return res;
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

	constexpr auto storeUnOpNames = mkUnOpNames(false);
	constexpr auto storeUnOpTraitNames = mkUnOpNames(true);
	constexpr auto unOpNames = nameArraysToSvs(storeUnOpNames);
	constexpr auto unOpTraitNames = nameArraysToSvs(storeUnOpTraitNames);
	
	constexpr auto storePostUnOpNames = mkPostUnOpNames(false);
	constexpr auto storePostUnOpTraitNames = mkPostUnOpNames(true);
	constexpr auto postUnOpNames = nameArraysToSvs(storePostUnOpNames);
	constexpr auto postUnOpTraitNames = nameArraysToSvs(storePostUnOpTraitNames);

	constexpr auto storeBinOpNames = mkPostBinOpNames(false);
	constexpr auto storeBinOpTraitNames = mkPostBinOpNames(true);
	constexpr auto binOpNames = nameArraysToSvs(storeBinOpNames);
	constexpr auto binOpTraitNames = nameArraysToSvs(storeBinOpTraitNames);


	constexpr char RANGE_OP_TRAIT_NAME[] = "Boundable";
}