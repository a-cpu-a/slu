/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <string>
#include <span>
#include <vector>

//https://www.lua.org/manual/5.4/manual.html
//https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form
//https://www.sciencedirect.com/topics/computer-science/backus-naur-form

#include <slu/Settings.hpp>
#include <slu/ext/CppMatch.hpp>
#include <slu/parser/Input.hpp>
#include <slu/parser/State.hpp>
#include <slu/parser/VecInput.hpp>
#include <slu/parser/basic/CharInfo.hpp>
#include <slu/paint/SemOutputStream.hpp>

namespace slu::paint
{
	template<AnySemOutput Se>
	inline void paintBinOp(Se& se, const parse::BinOpType& itm)
	{
		switch (itm)
		{
		case parse::BinOpType::ADD:
			paintKw<Tok::GEN_OP>(se, "+");
			break;
		case parse::BinOpType::SUBTRACT:
			paintKw<Tok::GEN_OP>(se, "-");
			break;
		case parse::BinOpType::MULTIPLY:
			paintKw<Tok::GEN_OP>(se, "*");
			break;
		case parse::BinOpType::DIVIDE:
			paintKw<Tok::GEN_OP>(se, "/");
			break;
		case parse::BinOpType::FLOOR_DIVIDE:
			paintKw<Tok::GEN_OP>(se, "//");
			break;
		case parse::BinOpType::MODULO:
			paintKw<Tok::GEN_OP>(se, "%");
			break;
		case parse::BinOpType::EXPONENT:
			paintKw<Tok::GEN_OP>(se, "^");
			break;
		case parse::BinOpType::BITWISE_AND:
			paintKw<Tok::GEN_OP>(se, "&");
			break;
		case parse::BinOpType::BITWISE_OR:
			paintKw<Tok::GEN_OP>(se, "|");
			break;
		case parse::BinOpType::BITWISE_XOR:
			paintKw<Tok::GEN_OP>(se, "~");
			break;
		case parse::BinOpType::SHIFT_LEFT:
			paintKw<Tok::GEN_OP>(se, "<<");
			break;
		case parse::BinOpType::SHIFT_RIGHT:
			paintKw<Tok::GEN_OP>(se, ">>");
			break;
		case parse::BinOpType::CONCATENATE:
			paintKw<Tok::GEN_OP>(se, parse::sel<Se>("..", "++"));
			break;
		case parse::BinOpType::LESS_THAN:
			paintKw<Tok::GEN_OP>(se, "<");
			break;
		case parse::BinOpType::LESS_EQUAL:
			paintKw<Tok::GEN_OP>(se, "<=");
			break;
		case parse::BinOpType::GREATER_THAN:
			paintKw<Tok::GEN_OP>(se, ">");
			break;
		case parse::BinOpType::GREATER_EQUAL:
			paintKw<Tok::GEN_OP>(se, ">=");
			break;
		case parse::BinOpType::EQUAL:
			paintKw<Tok::GEN_OP>(se, "==");
			break;
		case parse::BinOpType::NOT_EQUAL:
			paintKw<Tok::GEN_OP>(se, parse::sel<Se>("~=", "!="));
			break;
		case parse::BinOpType::LOGICAL_AND:
			paintKw<Tok::AND>(se, "and");
			break;
		case parse::BinOpType::LOGICAL_OR:
			paintKw<Tok::OR>(se, "or");
			break;
			//Slu:
		case parse::BinOpType::ARRAY_MUL:
			paintKw<Tok::ARRAY_MUL>(se, "**");
			break;
		case parse::BinOpType::RANGE_BETWEEN:
			paintKw<Tok::RANGE>(se, "..");
			break;
		case parse::BinOpType::NONE:
			break;
		}

	}
	template<AnySemOutput Se>
	inline void paintPostUnOp(Se& se, const parse::PostUnOpType& itm)
	{
		switch (itm)
		{
		case parse::PostUnOpType::RANGE_AFTER:
			paintKw<Tok::RANGE>(se, "..");
			break;
		case parse::PostUnOpType::DEREF:
			paintKw<Tok::DEREF>(se, ".*");
			break;
		case parse::PostUnOpType::PROPOGATE_ERR:
			paintKw<Tok::GEN_OP>(se, "?");
			break;
		case parse::PostUnOpType::NONE:
			break;
		}
	}
	template<AnySemOutput Se>
	inline void paintUnOpItem(Se& se, const parse::UnOpItem& itm)
	{
		switch (itm.type)
		{
		case parse::UnOpType::BITWISE_NOT:
			paintKw<Tok::GEN_OP>(se, "~");
			break;
		case parse::UnOpType::LOGICAL_NOT:
			paintKw<Tok::GEN_OP>(se, parse::sel<Se>("not", "!"));
			break;
		case parse::UnOpType::NEGATE:
			paintKw<Tok::GEN_OP>(se, "-");
			break;
		case parse::UnOpType::LENGTH:
			paintKw<Tok::GEN_OP>(se, "#");
			break;
		case parse::UnOpType::ALLOCATE:
			paintKw<Tok::GEN_OP>(se, "alloc");
			break;
		case parse::UnOpType::RANGE_BEFORE:
			paintKw<Tok::RANGE>(se, "..");
			break;
		case parse::UnOpType::MUT:
			paintKw<Tok::MUT>(se, "mut");
			break;
		case parse::UnOpType::TO_PTR_CONST:
			paintKw<Tok::PTR_CONST>(se, "*");
			paintKw<Tok::PTR_CONST>(se, "const");
			break;
		case parse::UnOpType::TO_PTR_MUT:
			paintKw<Tok::MUT>(se, "*");
			paintKw<Tok::MUT>(se, "mut");
			break;
		case parse::UnOpType::TO_REF:
			paintKw<Tok::REF>(se, "&");
			paintLifetime(se, itm.life);
			break;
		case parse::UnOpType::TO_REF_MUT:
			paintKw<Tok::MUT>(se, "&");
			paintLifetime(se, itm.life);
			paintKw<Tok::MUT>(se, "mut");
			break;
		case parse::UnOpType::NONE:
			break;
		}
	}
}