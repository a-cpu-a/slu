/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <string>
#include <unordered_map>
#include <variant>
#include <algorithm>
#include <numeric>

#include <slua/parser/State.hpp>
#include <slua/lang/BasicState.hpp>

namespace slua::mlvl
{
	enum class Assoc : uint8_t { LEFT, RIGHT };


	template<bool isLua>
	constexpr uint8_t precedence(parse::BinOpType op) {
		switch (op)
		{
		case parse::BinOpType::EXPONENT: return 90;

		case parse::BinOpType::MODULO: if constexpr (!isLua)return 70;

		case parse::BinOpType::DIVIDE:
		case parse::BinOpType::FLOOR_DIVIDE:
		case parse::BinOpType::MULTIPLY: return 80;


		case parse::BinOpType::SUBTRACT:
		case parse::BinOpType::ADD: return 60;

		case parse::BinOpType::SHIFT_LEFT: 
		case parse::BinOpType::SHIFT_RIGHT: return 50;

		case parse::BinOpType::BITWISE_OR:if constexpr (isLua)return 40;
		case parse::BinOpType::BITWISE_XOR:if constexpr (isLua)return 41;
		case parse::BinOpType::BITWISE_AND: return 42;

			//Slua
		case parse::BinOpType::RANGE_BETWEEN: if constexpr (!isLua)return 30;
			break;

			//Lua
		case parse::BinOpType::CONCATENATE:if constexpr (!isLua) return 4;
			return 55;//Lua, between +- and << >>

		case parse::BinOpType::GREATER_EQUAL:
		case parse::BinOpType::GREATER_THAN:
		case parse::BinOpType::LESS_EQUAL:
		case parse::BinOpType::LESS_THAN: if constexpr(!isLua)return 11;

		case parse::BinOpType::EQUAL:
		case parse::BinOpType::NOT_EQUAL: return 10;


		case parse::BinOpType::LOGICAL_AND: return 6;
		case parse::BinOpType::LOGICAL_OR: return 5;

			//Slua
		case parse::BinOpType::ARRAY_CONSTRUCT: if constexpr (!isLua)return 2;
			break;

		case parse::BinOpType::NONE:
			break;
		}
		Slua_panic("Unknown operator, no precedence<slua>(BinOpType) defined");
	}
	template<bool isLua>
	constexpr uint8_t precedence(const parse::UnOpItem& op) {
		switch (op.type)
		{
			//Slua
		case parse::UnOpType::RANGE_BEFORE:	if constexpr (!isLua)return 30;//same as range between
		case parse::UnOpType::ALLOCATE:if constexpr (!isLua)return 0;
		case parse::UnOpType::DEREF:if constexpr (!isLua)return 100;//above exponent
			break;
			//Lua
		case parse::UnOpType::LENGTH:        // "#"
		case parse::UnOpType::BITWISE_NOT:   // "~"
			if constexpr (!isLua)break;//Not in slua
		case parse::UnOpType::NEGATE:        // "-"
		case parse::UnOpType::LOGICAL_NOT:   // "not"
			return 85;//Between exponent and mul, div, ..
			//Slua
		case parse::UnOpType::TO_REF:			// "&"
		case parse::UnOpType::TO_REF_MUT:		// "&mut"
		case parse::UnOpType::TO_PTR_CONST:	// "*const"
		case parse::UnOpType::TO_PTR_MUT:		// "*mut"
			//Pseudo, only for type prefixes
		case parse::UnOpType::MUT:				// "mut"
			if constexpr (isLua)break;
			return 85;//Between exponent and mul, div, ..
			//
		case parse::UnOpType::NONE:
			break;
		}
		Slua_panic("Unknown operator, no precedence<slua>(UnOpItem) defined");
	}
	template<bool isLua>
	constexpr uint8_t precedence(parse::PostUnOpType op) {
		static_assert(!isLua);
		switch (op)
		{
		case parse::PostUnOpType::PROPOGATE_ERR: return 110;//above exponent and deref

		case parse::PostUnOpType::RANGE_AFTER: return 30;//same as range between

		case parse::PostUnOpType::NONE:
			break;
		}
		Slua_panic("Unknown operator, no precedence<slua>(PostUnOpType) defined");
	}

	template<bool isLua>
	constexpr Assoc associativity(parse::BinOpType op) {
		if constexpr (isLua)
			return op == parse::BinOpType::EXPONENT ? Assoc::RIGHT : Assoc::LEFT;

		switch (op)
		{
		case parse::BinOpType::EXPONENT:
		case parse::BinOpType::SHIFT_LEFT:
		case parse::BinOpType::SHIFT_RIGHT:
		case parse::BinOpType::ARRAY_CONSTRUCT:
			return Assoc::RIGHT;
		default:
			return Assoc::LEFT;
		}
	}

	enum class OpKind : uint8_t { BinOp, UnOp, PostUnOp };
	struct MultiOpOrderEntry
	{
		size_t index;
		size_t opIdx;
		OpKind kind;
		uint8_t precedence;
		Assoc assoc = Assoc::LEFT; // Only relevant for BinOp
	};
	// Returns the order of operations as indices into `extra`
	// Store parse::BinOpType in `extra[...].first`
	// Store std::vector<parse::UnOpItem> in `extra[...].second.unOps`
	// Store std::vector<parse::PostUnOpType> in `extra[...].second.postUnOps`
	template<bool isLua>
	constexpr std::vector<MultiOpOrderEntry> multiOpOrder(const auto& m)
	{
		std::vector<MultiOpOrderEntry> ops;

		//First entry
		size_t j = 0;
		for (const auto& un : m.first.unOps)
		{
			ops.push_back({ 0,j++, OpKind::UnOp, precedence<isLua>(un),Assoc::RIGHT });
		}
		j = 0;
		for (const auto post : m.first.postUnOps)
		{
			ops.push_back({ 0,j++, OpKind::PostUnOp, precedence<isLua>(post),Assoc::RIGHT });
		}
		// Add binary ops
		for (size_t i = 0; i < m.extra.size(); ++i)
		{
			auto& bin = m.extra[i].first;
			ops.push_back({ i,0, OpKind::BinOp, precedence<isLua>(bin), associativity<isLua>(bin) });
			j = 0;
			for (const auto& un : m.extra[i].second.unOps)
			{
				ops.push_back({ i,j++, OpKind::UnOp, precedence<isLua>(un),Assoc::RIGHT });
			}
			j = 0;
			for (const auto post : m.extra[i].second.postUnOps)
			{
				ops.push_back({ i,j++, OpKind::PostUnOp, precedence<isLua>(post),Assoc::RIGHT });
			}
		}

		std::sort(ops.begin(), ops.end(), [](const MultiOpOrderEntry& a, const MultiOpOrderEntry& b) {

			if (a.index == b.index)
			{
				if (a.kind == OpKind::BinOp || b.kind == OpKind::BinOp)
				{//Only one is binOp
					if (a.kind == OpKind::BinOp)
						return a.precedence > b.precedence;

					//b is bin
					return a.kind == OpKind::BinOp;
				}
				if (a.kind == OpKind::UnOp && b.kind == OpKind::UnOp)
					return a.precedence > b.precedence;
				if (a.kind == OpKind::PostUnOp && b.kind == OpKind::PostUnOp)
					return a.precedence > b.precedence;
			}

			if (a.precedence != b.precedence)
				return a.precedence > b.precedence;

			return (a.assoc == Assoc::LEFT) ? a.index < b.index : a.index > b.index;
		});
		return ops;
	}

	struct UnOpOrderEntry
	{
		size_t index;
		bool isPost; // false = unOp, true = postUnOp
		uint8_t precedence;
	};
	template<bool isLua>
	constexpr std::vector<size_t> unaryOpOrder(const auto& expr) 
	{
		std::vector<UnOpOrderEntry> ops;

		for (size_t i = 0; i < expr.unOps.size(); ++i)
		{
			ops.push_back({ i, false, precedence<isLua>(expr.unOps[i]) });
		}
		for (size_t i = 0; i < expr.postUnOps.size(); ++i)
		{
			ops.push_back({ i, true, precedence<isLua>(expr.postUnOps[i]) });
		}

		std::sort(ops.begin(), ops.end(), [](const UnOpOrderEntry& a, const UnOpOrderEntry& b) {
			return a.precedence > b.precedence; // Higher precedence first
		});
		return ops;
	}
}