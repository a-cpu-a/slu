/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <string>
#include <unordered_map>
#include <variant>
#include <algorithm>
#include <numeric>

#include <slu/parser/State.hpp>
#include <slu/lang/BasicState.hpp>

namespace slu::mlvl
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

			//Slu
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

			//Slu
		case parse::BinOpType::ARRAY_MUL: if constexpr (!isLua)return 2;
			break;

		case parse::BinOpType::NONE:
			break;
		}
		Slu_panic("Unknown operator, no precedence<slu>(BinOpType) defined");
	}
	template<bool isLua>
	constexpr uint8_t precedence(const parse::UnOpItem& op) {
		switch (op.type)
		{
			//Slu
		case parse::UnOpType::RANGE_BEFORE:	if constexpr (!isLua)return 30;//same as range between
		case parse::UnOpType::ALLOCATE:if constexpr (!isLua)return 0;
			break;
			//Lua
		case parse::UnOpType::LENGTH:        // "#"
		case parse::UnOpType::BITWISE_NOT:   // "~"
			if constexpr (!isLua)break;//Not in slu
		case parse::UnOpType::NEGATE:        // "-"
		case parse::UnOpType::LOGICAL_NOT:   // "not"
			return 85;//Between exponent and mul, div, ..
			//Slu
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
		Slu_panic("Unknown operator, no precedence<slu>(UnOpItem) defined");
	}
	template<bool isLua>
	constexpr uint8_t precedence(parse::PostUnOpType op) {
		static_assert(!isLua);
		switch (op)
		{
		case parse::PostUnOpType::PROPOGATE_ERR:
		case parse::PostUnOpType::DEREF:return 100;//above exponent

		case parse::PostUnOpType::RANGE_AFTER: return 30;//same as range between

		case parse::PostUnOpType::NONE:
			break;
		}
		Slu_panic("Unknown operator, no precedence<slu>(PostUnOpType) defined");
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
		Assoc assoc = Assoc::LEFT; // Only varying for BinOp
	};
	struct ExprUnOpsEntry
	{
		size_t preConsumed=0;
		size_t sufConsumed=0;
	};

	template<bool isLua>
	constexpr uint8_t calcPrecedence(auto p, const auto& end)
	{
		uint8_t prec = 0;
		while (p!=end)
		{
			prec = std::max(prec, precedence<isLua>(*p));
			p++;
		}
		return prec;
	}

	template<bool isLua>
	constexpr void consumeUnOps(std::vector<MultiOpOrderEntry>& unOps,const auto& item,const size_t itemIdx, ExprUnOpsEntry& entry,const bool onLeftSide,const uint8_t minPrecedence)
	{
		auto pSuf = item.postUnOps.cbegin()+ entry.sufConsumed;
		auto pPre = item.unOps.crbegin() + entry.preConsumed;
		//Loop pre in reverse, to get it as inner->outter

		bool allowLeft = true;
		bool allowRight = true;

		while (pSuf != item.postUnOps.cend() || pPre != item.unOps.crend())
		{
			const bool preAlive = pPre != item.unOps.crend();
			const bool sufAlive = pSuf != item.postUnOps.cend();

			const uint8_t prePrec = preAlive ? calcPrecedence<isLua>(pPre, item.unOps.crend()) : 0;
			const uint8_t sufPrec = sufAlive ? calcPrecedence<isLua>(pSuf, item.postUnOps.cend()) : 0;

			if (sufAlive && (!preAlive || sufPrec > prePrec))
			{
				//Once its not allowed once, it never will be, ever again.
				allowLeft = allowLeft && (onLeftSide || sufPrec > minPrecedence);
				if (allowLeft)
				{
					unOps.push_back({ itemIdx,entry.sufConsumed++, OpKind::PostUnOp, sufPrec,Assoc::LEFT });
				}
				pSuf++;
			}
			else
			{
				allowRight = allowRight && (!onLeftSide || prePrec > minPrecedence);
				if (allowRight)
				{
					unOps.push_back({ itemIdx,entry.preConsumed++, OpKind::UnOp, prePrec,Assoc::RIGHT });
				}
				pPre++;
			}
		}
	}

	// Returns the order of operations as indices into `extra`
	// Store parse::BinOpType in `extra[...].first`
	// Store std::vector<parse::UnOpItem> in `extra[...].second.unOps`
	// Store std::vector<parse::PostUnOpType> in `extra[...].second.postUnOps`
	template<bool isLua>
	constexpr std::vector<MultiOpOrderEntry> multiOpOrder(const auto& m)
	{
		/*

		When sorting, binop to binop order is always the same, even when removing all unary ops.
		This means you can sort the un ops after bin-ops.

		Sorting un ops:

		left > right, if both have same precedence.
		UnOp > PostUnOp, if both have same precedence.


		Un ops can only apply to their expression, or the result of a bin op.

		If you have a un op with a higher precedence than a bin op, it will always be applied before even if there are lower un ops before it.
		With equal precedences, the un ops are applied after.
		Any un ops in between a bin op are applied before the bin op.

		un op precedences between other un ops only matters if they are on different sides of a expr.
		This means that un ops will expand outwards from the expression, going to the highest precedence un op, until matching the precedence of a bin op.

		The effective precedence of a un op is the highest of any that are applied after, and itself.
		(*-0) -> negative takes the max of (-,*) sees that * has more, so copies it.
		*/

		std::vector<MultiOpOrderEntry> ops;

		//First entry
		/*size_t j = 0;
		for (const auto& un : m.first.unOps)
		{
			ops.push_back({ 0,j++, OpKind::UnOp, precedence<isLua>(un),Assoc::RIGHT });
		}
		j = 0;
		for (const auto post : m.first.postUnOps)
		{
			ops.push_back({ 0,j++, OpKind::PostUnOp, precedence<isLua>(post),Assoc::LEFT });
		}*/
		// Add binary ops
		for (size_t i = 0; i < m.extra.size(); ++i)
		{
			auto& bin = m.extra[i].first;
			ops.push_back({ i,0, OpKind::BinOp, precedence<isLua>(bin), associativity<isLua>(bin) });
			/*j = 0;
			for (const auto& un : m.extra[i].second.unOps)
			{
				ops.push_back({ i,j++, OpKind::UnOp, precedence<isLua>(un),Assoc::RIGHT });
			}
			j = 0;
			for (const auto post : m.extra[i].second.postUnOps)
			{
				ops.push_back({ i,j++, OpKind::PostUnOp, precedence<isLua>(post),Assoc::LEFT });
			}*/
		}

		std::sort(ops.begin(), ops.end(), [](const MultiOpOrderEntry& a, const MultiOpOrderEntry& b) {

			if (a.precedence != b.precedence)
				return a.precedence > b.precedence;

			return (a.assoc == Assoc::LEFT) ? a.index < b.index : a.index > b.index;
		});
		
		std::vector<ExprUnOpsEntry> exprUnOps(ops.size()+1);
		std::vector<std::vector<MultiOpOrderEntry>> unOps(ops.size());
		std::vector<MultiOpOrderEntry> unOpsLast;
		size_t unOpCount = m.first.unOps.size() + m.first.postUnOps.size();
		

		for (size_t i = 0; i < ops.size(); i++)
		{
			const MultiOpOrderEntry& e = ops[i];
			
			ExprUnOpsEntry& leftEntry = exprUnOps[e.index];
			const auto& left = (e.index==0)
				? m.first
				: m.extra[e.index-1].second;
			ExprUnOpsEntry& rightEntry = exprUnOps[e.index+1];
			const auto& right = m.extra[e.index].second;

			//e.index is unique to every binop.
			unOpCount += right.unOps.size() + right.postUnOps.size();

			const bool lAssoc = e.assoc == Assoc::LEFT;

			const auto& item1 = lAssoc ? left : right;
			const size_t item1Idx = lAssoc ? e.index : e.index + 1; // +1, cuz 1 is first expr
			const auto& item2 = lAssoc ? right : left;
			const size_t item2Idx = lAssoc ? e.index + 1 : e.index;
			ExprUnOpsEntry& ent1 = lAssoc ? leftEntry : rightEntry;
			ExprUnOpsEntry& ent2 = lAssoc ? rightEntry : leftEntry;

			consumeUnOps<isLua>(unOps[i],item1, item1Idx, ent1, lAssoc,e.precedence);
			consumeUnOps<isLua>(unOps[i],item2, item2Idx, ent2,!lAssoc,e.precedence);
		}
		//Consume first, last expr un ops if needed.
		// min op prec is 0xFF, to mark the opposite sides as not usable.
		consumeUnOps<isLua>(unOpsLast, m.first,0, exprUnOps.front(), false, 0xFF);
		consumeUnOps<isLua>(unOpsLast, m.extra.back().second,m.extra.size(), exprUnOps.back(), true, 0xFF);

		/*
		size_t j = 0;
		for (const auto& un : m.first.unOps)
		{
			unOps.push_back({ 0,j++,b, OpKind::UnOp, precedence<isLua>(un),Assoc::RIGHT });
		}
		j = 0;
		for (const auto post : m.first.postUnOps)
		{
			unOps.push_back({ 0,j++,b, OpKind::PostUnOp, precedence<isLua>(post),Assoc::LEFT });
		}*/

			/*
			j = 0;
			for (const auto& un : m.extra[i.index].second.unOps)
			{
				unOps.push_back({ i.index,j++,b, OpKind::UnOp, precedence<isLua>(un),Assoc::RIGHT });
			}
			j = 0;
			for (const auto post : m.extra[i.index].second.postUnOps)
			{
				unOps.push_back({ i.index,j++,b, OpKind::PostUnOp, precedence<isLua>(post),Assoc::LEFT });
			}*/

		std::vector<MultiOpOrderEntry> opsRes;
		opsRes.reserve(ops.size() + unOpCount);

		size_t j = 0;
		for (std::vector<MultiOpOrderEntry>& i : unOps)
		{
			std::move(i.begin(), i.end(), std::back_inserter(opsRes));
			opsRes.emplace_back(std::move(ops[j++]));
		}
		std::move(unOpsLast.begin(), unOpsLast.end(), std::back_inserter(opsRes));

		_ASSERT(opsRes.size() == ops.size() + unOpCount);

		return opsRes;
	}

	struct UnOpOrderEntry
	{
		size_t index;
		bool isPost; // false = unOp, true = postUnOp
		uint8_t precedence;
	};
	template<bool isLua>
	constexpr std::vector<size_t> unaryOpOrderTodo(const auto& expr) 
	{
		//TODO: this doesnt handle mixed order on one side, assoc of both.
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