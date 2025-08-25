/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <cstdint>
#include <array>
#include <string_view>
import slu.ast.enums;
#include <slu/parser/basic/CharInfo.hpp>

namespace slu::parse
{
	constexpr char RANGE_OP_TRAIT_NAME[] = "Boundable";

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
	constexpr void rewriteNameArrayElement(auto res, auto idx, const char* str)
	{
		size_t i = 0;
		while (str[i] != 0)
		{
			res[(size_t)idx - 1][i] = str[i];
			i++;
		}
	}
	constexpr auto mkUnOpNames(bool forTrait)
	{
		std::array<std::array<char,10>, (size_t)UnOpType::ENUM_SIZE> res = {
			"neg","not",

			"rangeMax\0",//!!!

			"_alloc",
			"ref", "refMut", "refConst","refShare",
			"ptr", "ptrMut", "ptrConst","ptrShare",
			"markMut"
		};
		handleTraitCase(res,forTrait);
		if (forTrait)
		{
			rewriteNameArrayElement(res, UnOpType::RANGE_BEFORE, RANGE_OP_TRAIT_NAME);
		}
		return res;
	}
	constexpr auto mkPostUnOpNames(bool forTrait)
	{
		std::array<std::array<char,10>, (size_t)PostUnOpType::ENUM_SIZE> res = {
			"rangeMin\0",//!!!
			"deref", "_try"
		};
		handleTraitCase(res,forTrait);
		if (forTrait)
		{
			rewriteNameArrayElement(res, PostUnOpType::RANGE_AFTER, RANGE_OP_TRAIT_NAME);
		}
		return res;
	}
	constexpr auto mkBinOpNames(bool forTrait)
	{
		std::array<std::array<char,14>, (size_t)BinOpType::ENUM_SIZE> res = {
			"add","sub", "mul", "div","flrDiv", "pow", "rem",
			"bitAnd", "bitXor", "bitOr", "shr", "shl",
			"concat",

			"lt\0\0\0\0\0\0\0\0", "le\0\0\0\0\0\0\0\0", //!!!
			"gt\0\0\0\0\0\0\0\0", "ge\0\0\0\0\0\0\0\0", //!!!
			"eq\0\0\0\0\0\0\0",   "ne\0\0\0\0\0\0\0",	//!!!
			"and","or",//!!!
			
			"rep",
			"range\0\0\0\0",//!!!
			"_mkResult",
			"_union",
			"as\0\0\0\0"//!!! // the \0's are to work around ?msvc? stuff.
		};
		handleTraitCase(res,forTrait);
		if (forTrait)
		{
			rewriteNameArrayElement(res,BinOpType::RANGE_BETWEEN, RANGE_OP_TRAIT_NAME);
			auto ltgtTraitName = "PartialOrd";
			rewriteNameArrayElement(res, BinOpType::LESS_THAN, ltgtTraitName);
			rewriteNameArrayElement(res, BinOpType::GREATER_THAN, ltgtTraitName);
			rewriteNameArrayElement(res, BinOpType::LESS_EQUAL, ltgtTraitName);
			rewriteNameArrayElement(res, BinOpType::GREATER_EQUAL, ltgtTraitName);
			auto eqTraitName = "PartialEq";
			rewriteNameArrayElement(res, BinOpType::EQUAL, eqTraitName);
			rewriteNameArrayElement(res, BinOpType::NOT_EQUAL, eqTraitName);
		}
		else
		{
			auto& asStr = res[((size_t)BinOpType::AS) - 1];
			asStr[2] = 'T';
			asStr[3] = 'y';
			asStr[4] = 'p';
			asStr[5] = 'e';
		}

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

	constexpr auto storeBinOpNames = mkBinOpNames(false);
	constexpr auto storeBinOpTraitNames = mkBinOpNames(true);
	constexpr auto binOpNames = nameArraysToSvs(storeBinOpNames);
	constexpr auto binOpTraitNames = nameArraysToSvs(storeBinOpTraitNames);
}