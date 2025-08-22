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


	constexpr uint8_t precedence(parse::BinOpType op) {
		switch (op)
		{
		case parse::BinOpType::EXPONENT: return 90;

		case parse::BinOpType::MODULO: return 70;

		case parse::BinOpType::DIVIDE:
		case parse::BinOpType::FLOOR_DIVIDE:
		case parse::BinOpType::MULTIPLY: return 80;


		case parse::BinOpType::SUBTRACT:
		case parse::BinOpType::ADD: return 60;

		case parse::BinOpType::SHIFT_LEFT: 
		case parse::BinOpType::SHIFT_RIGHT: return 50;

		case parse::BinOpType::BITWISE_OR:
		case parse::BinOpType::BITWISE_XOR:
		case parse::BinOpType::BITWISE_AND: return 42;

		case parse::BinOpType::RANGE_BETWEEN: return 30;

		case parse::BinOpType::CONCATENATE: return 20;

		case parse::BinOpType::GREATER_EQUAL:
		case parse::BinOpType::GREATER_THAN:
		case parse::BinOpType::LESS_EQUAL:
		case parse::BinOpType::LESS_THAN: return 11;

		case parse::BinOpType::EQUAL:
		case parse::BinOpType::NOT_EQUAL: return 10;


		case parse::BinOpType::LOGICAL_AND: return 6;
		case parse::BinOpType::LOGICAL_OR: return 5;

		case parse::BinOpType::ARRAY_MUL: return 21;
		case parse::BinOpType::MAKE_RESULT: return 15;
		case parse::BinOpType::UNION: return 17;
		case parse::BinOpType::AS: return 13;

		case parse::BinOpType::NONE:
			break;
		}
		Slu_panic("Unknown operator, no precedence<slu>(BinOpType) defined");
	}
	constexpr uint8_t precedence(const parse::UnOpItem& op) {
		switch (op.type)
		{
			//Slu
		case parse::UnOpType::RANGE_BEFORE:	return 30;//same as range between
		case parse::UnOpType::ALLOCATE:return 0;
			break;
		case parse::UnOpType::NEGATE:        // "-"
		case parse::UnOpType::LOGICAL_NOT:   // "not"
			return 85;//Between exponent and mul, div, ..
		case parse::UnOpType::TO_REF:			// "&"
		case parse::UnOpType::TO_REF_CONST:		// "&const"
		case parse::UnOpType::TO_REF_SHARE:		// "&share"
		case parse::UnOpType::TO_REF_MUT:		// "&mut"
		case parse::UnOpType::TO_PTR:			// "*"
		case parse::UnOpType::TO_PTR_CONST:		// "*const"
		case parse::UnOpType::TO_PTR_SHARE:		// "*share"
		case parse::UnOpType::TO_PTR_MUT:		// "*mut"
			//Pseudo, only for type prefixes
		case parse::UnOpType::MUT:				// "mut"
			return 85;//Between exponent and mul, div, ..
			//
		case parse::UnOpType::NONE:
			break;
		}
		Slu_panic("Unknown operator, no precedence<slu>(UnOpItem) defined");
	}
	constexpr uint8_t precedence(parse::PostUnOpType op) {
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

	constexpr Assoc associativity(parse::BinOpType op) {
		switch (op)
		{
		case parse::BinOpType::EXPONENT:
		case parse::BinOpType::SHIFT_LEFT:
		case parse::BinOpType::SHIFT_RIGHT:
		case parse::BinOpType::ARRAY_MUL:
			return Assoc::RIGHT;
		default:
			return Assoc::LEFT;
		}
	}

	enum class OpKind : uint8_t { BinOp, UnOp, PostUnOp,Expr };
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
		size_t preConsumed = 0;
		size_t sufConsumed : 63 = 0;
		size_t used = false;
	};

	constexpr uint8_t calcPrecedence(auto p, const auto& end)
	{
		uint8_t prec = 0;
		while (p!=end)
		{
			prec = std::max(prec, precedence(*p));
			p++;
		}
		return prec;
	}

	constexpr void consumeUnOps(std::vector<MultiOpOrderEntry>& unOps,const auto& item,const size_t itemIdx, ExprUnOpsEntry& entry,const bool onLeftSide,const uint8_t minPrecedence)
	{
		if (!entry.used)
		{
			unOps.insert(unOps.end(), MultiOpOrderEntry{.index = itemIdx,.kind = OpKind::Expr});
			entry.used = true;
		}

		auto pSuf = item.postUnOps.cbegin() + entry.sufConsumed;
		auto pPre = item.unOps.crbegin() + entry.preConsumed;
		//Loop pre in reverse, to get it as inner->outter

		bool allowLeft = true;
		bool allowRight = true;

		while (pSuf != item.postUnOps.cend() || pPre != item.unOps.crend())
		{
			const bool preAlive = pPre != item.unOps.crend();
			const bool sufAlive = pSuf != item.postUnOps.cend();

			const uint8_t prePrec = preAlive ? calcPrecedence(pPre, item.unOps.crend()) : 0;
			const uint8_t sufPrec = sufAlive ? calcPrecedence(pSuf, item.postUnOps.cend()) : 0;

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

		// Add binary ops
		for (size_t i = 0; i < m.extra.size(); ++i)
		{
			auto& bin = m.extra[i].first;
			ops.push_back({ i+1,0, OpKind::BinOp, precedence(bin), associativity(bin) });
		}

		std::sort(ops.begin(), ops.end(), [](const MultiOpOrderEntry& a, const MultiOpOrderEntry& b) {

			if (a.precedence != b.precedence)
				return a.precedence > b.precedence;

			return (a.assoc == Assoc::LEFT) ? a.index < b.index : a.index > b.index;
		});
		
		std::vector<ExprUnOpsEntry> exprUnOps(ops.size()+1);
		std::vector<std::vector<MultiOpOrderEntry>> unOps(ops.size());
		std::vector<MultiOpOrderEntry> unOpsLast;
		size_t unOpCount = m.first->unOps.size() + m.first->postUnOps.size();
		

		for (size_t i = 0; i < ops.size(); i++)
		{
			const MultiOpOrderEntry& e = ops[i];
			
			ExprUnOpsEntry& leftEntry = exprUnOps[e.index-1];
			const auto& left = (e.index==1)
				? *m.first
				: m.extra[e.index-2].second;
			ExprUnOpsEntry& rightEntry = exprUnOps[e.index];
			const auto& right = m.extra[e.index-1].second;

			//e.index is unique to every binop.
			unOpCount += right.unOps.size() + right.postUnOps.size();

			const bool lAssoc = e.assoc == Assoc::LEFT;

			const auto& item1 = lAssoc ? left : right;
			const size_t item1Idx = lAssoc ? e.index-1 : e.index ; // -1, cuz 1 is first expr
			const auto& item2 = lAssoc ? right : left;
			const size_t item2Idx = lAssoc ? e.index : e.index-1;
			ExprUnOpsEntry& ent1 = lAssoc ? leftEntry : rightEntry;
			ExprUnOpsEntry& ent2 = lAssoc ? rightEntry : leftEntry;

			consumeUnOps(unOps[i],item1, item1Idx, ent1, lAssoc,e.precedence);
			consumeUnOps(unOps[i],item2, item2Idx, ent2,!lAssoc,e.precedence);
		}
		//Consume first, last expr un ops if needed.
		// min op prec is 0xFF, to mark the opposite sides as not usable.
		consumeUnOps(unOpsLast, *m.first,0, exprUnOps.front(), false, 0xFF);
		consumeUnOps(unOpsLast, m.extra.back().second,m.extra.size(), exprUnOps.back(), true, 0xFF);

		std::vector<MultiOpOrderEntry> opsRes;
		opsRes.reserve(ops.size() + unOpCount);

		size_t j = 0;
		for (std::vector<MultiOpOrderEntry>& i : unOps)
		{
			std::move(i.begin(), i.end(), std::back_inserter(opsRes));
			opsRes.emplace_back(std::move(ops[j++]));
		}
		std::move(unOpsLast.begin(), unOpsLast.end(), std::back_inserter(opsRes));

		_ASSERT(opsRes.size() == ops.size() + unOpCount+1+m.extra.size());

		return opsRes;
	}

	/// @returns the order of unary operations as a vector of bools, where true means post-unary operation
	constexpr std::vector<bool> unaryOpOrder(const parse::ExprV<true>& expr)
	{
		size_t opCount = expr.postUnOps.size() + expr.unOps.size();
		if(expr.postUnOps.empty() || expr.unOps.empty())
		{// Only one side, so no order.
			std::vector<bool> res(opCount, expr.unOps.empty());
			return res;
		}

		std::vector<bool> res(opCount,false);
		size_t unOpIdx = expr.unOps.size()-1;
		size_t postUnOpIdx = 0;

		uint8_t unPrec = 0;
		uint8_t postPrec = 0;

		for (size_t i = 0; i < res.size(); i++)
		{
			if(precedence(expr.unOps[unOpIdx]) >= precedence(expr.postUnOps.at(postUnOpIdx)))
			{
				res[i] = false; // UnOp
				unOpIdx--;
				if (unOpIdx == SIZE_MAX)
				{// Fill the rest.
					std::fill_n(res.begin() + i + 1, res.size() - i - 1, true);
					break;
				}
			}
			else
			{
				res[i] = true; // PostUnOp
				postUnOpIdx++;
				if(postUnOpIdx >= expr.postUnOps.size())
				{// Fill the rest.
					std::fill_n(res.begin() + i + 1, res.size() - i - 1, false);
					break;
				}
			}
		}
		return res;
	}
}