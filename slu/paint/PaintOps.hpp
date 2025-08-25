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
	inline void paintBinOp(Se& se, const ast::BinOpType& itm)
	{
		switch (itm)
		{
		case ast::BinOpType::ADD:
			paintKw<Tok::GEN_OP>(se, "+");
			break;
		case ast::BinOpType::SUBTRACT:
			paintKw<Tok::GEN_OP>(se, "-");
			break;
		case ast::BinOpType::MULTIPLY:
			paintKw<Tok::GEN_OP>(se, "*");
			break;
		case ast::BinOpType::DIVIDE:
			paintKw<Tok::GEN_OP>(se, "/");
			break;
		case ast::BinOpType::FLOOR_DIVIDE:
			paintKw<Tok::GEN_OP>(se, "//");
			break;
		case ast::BinOpType::MODULO:
			paintKw<Tok::GEN_OP>(se, "%");
			break;
		case ast::BinOpType::EXPONENT:
			paintKw<Tok::GEN_OP>(se, "^");
			break;
		case ast::BinOpType::BITWISE_AND:
			paintKw<Tok::GEN_OP>(se, "&");
			break;
		case ast::BinOpType::BITWISE_OR:
			paintKw<Tok::GEN_OP>(se, "|");
			break;
		case ast::BinOpType::BITWISE_XOR:
			paintKw<Tok::GEN_OP>(se, "~");
			break;
		case ast::BinOpType::SHIFT_LEFT:
			paintKw<Tok::GEN_OP>(se, "<<");
			break;
		case ast::BinOpType::SHIFT_RIGHT:
			paintKw<Tok::GEN_OP>(se, ">>");
			break;
		case ast::BinOpType::CONCATENATE:
			paintKw<Tok::GEN_OP>(se, "++");
			break;
		case ast::BinOpType::LESS_THAN:
			paintKw<Tok::GEN_OP>(se, "<");
			break;
		case ast::BinOpType::LESS_EQUAL:
			paintKw<Tok::GEN_OP>(se, "<=");
			break;
		case ast::BinOpType::GREATER_THAN:
			paintKw<Tok::GEN_OP>(se, ">");
			break;
		case ast::BinOpType::GREATER_EQUAL:
			paintKw<Tok::GEN_OP>(se, ">=");
			break;
		case ast::BinOpType::EQUAL:
			paintKw<Tok::GEN_OP>(se, "==");
			break;
		case ast::BinOpType::NOT_EQUAL:
			paintKw<Tok::GEN_OP>(se, "!=");
			break;
		case ast::BinOpType::LOGICAL_AND:
			paintKw<Tok::AND>(se, "and");
			break;
		case ast::BinOpType::LOGICAL_OR:
			paintKw<Tok::OR>(se, "or");
			break;
			//Slu:
		case ast::BinOpType::ARRAY_MUL:
			paintKw<Tok::ARRAY_MUL>(se, "**");
			break;
		case ast::BinOpType::RANGE_BETWEEN:
			paintKw<Tok::RANGE>(se, "..");
			break;
		case ast::BinOpType::MAKE_RESULT:
			paintKw<Tok::GEN_OP>(se, "~~");
			break;
		case ast::BinOpType::UNION:
			paintKw<Tok::GEN_OP>(se, "||");
			break;
		case ast::BinOpType::AS:
			paintKw<Tok::AS>(se, "as");
			break;
		case ast::BinOpType::NONE:
			break;
		}

	}
	template<AnySemOutput Se>
	inline void paintPostUnOp(Se& se, const ast::PostUnOpType& itm)
	{
		switch (itm)
		{
		case ast::PostUnOpType::RANGE_AFTER:
			paintKw<Tok::RANGE>(se, "..");
			break;
		case ast::PostUnOpType::DEREF:
			paintKw<Tok::DEREF>(se, ".*");
			break;
		case ast::PostUnOpType::PROPOGATE_ERR:
			paintKw<Tok::GEN_OP>(se, "?");
			break;
		case ast::PostUnOpType::NONE:
			break;
		}
	}
	template<AnySemOutput Se>
	inline void paintUnOpItem(Se& se, const parse::UnOpItem& itm)
	{
		switch (itm.type)
		{
		case ast::UnOpType::LOGICAL_NOT:
			paintKw<Tok::GEN_OP>(se, "!");
			break;
		case ast::UnOpType::NEGATE:
			paintKw<Tok::GEN_OP>(se, "-");
			break;
		case ast::UnOpType::ALLOCATE:
			paintKw<Tok::GEN_OP>(se, "alloc");
			break;
		case ast::UnOpType::RANGE_BEFORE:
			paintKw<Tok::RANGE>(se, "..");
			break;
		case ast::UnOpType::MUT:
			paintKw<Tok::MUT>(se, "mut");
			break;

		case ast::UnOpType::TO_PTR:
			paintKw<Tok::PTR>(se, "*");
			break;
		case ast::UnOpType::TO_PTR_CONST:
			paintKw<Tok::PTR_CONST>(se, "*");
			paintKw<Tok::PTR_CONST>(se, "const");
			break;
		case ast::UnOpType::TO_PTR_MUT:
			paintKw<Tok::PTR_MUT>(se, "*");
			paintKw<Tok::PTR_MUT>(se, "mut");
			break;
		case ast::UnOpType::TO_PTR_SHARE:
			paintKw<Tok::PTR_SHARE>(se, "*");
			paintKw<Tok::PTR_SHARE>(se, "share");
			break;

		case ast::UnOpType::TO_REF:
			paintKw<Tok::REF>(se, "&");
			paintLifetime(se, itm.life);
			break;
		case ast::UnOpType::TO_REF_MUT:
			paintKw<Tok::REF_MUT>(se, "&");
			paintLifetime(se, itm.life);
			paintKw<Tok::REF_MUT>(se, "mut");
			break;
		case ast::UnOpType::TO_REF_CONST:
			paintKw<Tok::REF_CONST>(se, "&");
			paintLifetime(se, itm.life);
			paintKw<Tok::REF_CONST>(se, "const");
			break;
		case ast::UnOpType::TO_REF_SHARE:
			paintKw<Tok::REF_SHARE>(se, "&");
			paintLifetime(se, itm.life);
			paintKw<Tok::REF_SHARE>(se, "share");
			break;
		case ast::UnOpType::NONE:
			break;
		}
	}
}