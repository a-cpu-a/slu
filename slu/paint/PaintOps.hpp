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

#include <slu/ext/CppMatch.hpp>
import slu.ast.state;
import slu.parse.input;
#include <slu/parse/VecInput.hpp>
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
		case ast::BinOpType::SUB:
			paintKw<Tok::GEN_OP>(se, "-");
			break;
		case ast::BinOpType::MUL:
			paintKw<Tok::GEN_OP>(se, "*");
			break;
		case ast::BinOpType::DIV:
			paintKw<Tok::GEN_OP>(se, "/");
			break;
		case ast::BinOpType::FLOOR_DIV:
			paintKw<Tok::GEN_OP>(se, "//");
			break;
		case ast::BinOpType::REM:
			paintKw<Tok::GEN_OP>(se, "%");
			break;
		case ast::BinOpType::EXP:
			paintKw<Tok::GEN_OP>(se, "^");
			break;
		case ast::BinOpType::BIT_AND:
			paintKw<Tok::GEN_OP>(se, "&");
			break;
		case ast::BinOpType::BIT_OR:
			paintKw<Tok::GEN_OP>(se, "|");
			break;
		case ast::BinOpType::BIT_XOR:
			paintKw<Tok::GEN_OP>(se, "~");
			break;
		case ast::BinOpType::SHL:
			paintKw<Tok::GEN_OP>(se, "<<");
			break;
		case ast::BinOpType::SHR:
			paintKw<Tok::GEN_OP>(se, ">>");
			break;
		case ast::BinOpType::CONCAT:
			paintKw<Tok::GEN_OP>(se, "++");
			break;
		case ast::BinOpType::LT:
			paintKw<Tok::GEN_OP>(se, "<");
			break;
		case ast::BinOpType::LE:
			paintKw<Tok::GEN_OP>(se, "<=");
			break;
		case ast::BinOpType::GT:
			paintKw<Tok::GEN_OP>(se, ">");
			break;
		case ast::BinOpType::GE:
			paintKw<Tok::GEN_OP>(se, ">=");
			break;
		case ast::BinOpType::EQ:
			paintKw<Tok::GEN_OP>(se, "==");
			break;
		case ast::BinOpType::NE:
			paintKw<Tok::GEN_OP>(se, "!=");
			break;
		case ast::BinOpType::AND:
			paintKw<Tok::AND>(se, "and");
			break;
		case ast::BinOpType::OR:
			paintKw<Tok::OR>(se, "or");
			break;
			//Slu:
		case ast::BinOpType::REP:
			paintKw<Tok::REP>(se, "**");
			break;
		case ast::BinOpType::RANGE:
			paintKw<Tok::RANGE>(se, "..");
			break;
		case ast::BinOpType::MK_RESULT:
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
		case ast::PostUnOpType::TRY:
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
		case ast::UnOpType::NOT:
			paintKw<Tok::GEN_OP>(se, "!");
			break;
		case ast::UnOpType::NEG:
			paintKw<Tok::GEN_OP>(se, "-");
			break;
		case ast::UnOpType::ALLOC:
			paintKw<Tok::GEN_OP>(se, "alloc");
			break;
		case ast::UnOpType::RANGE_BEFORE:
			paintKw<Tok::RANGE>(se, "..");
			break;
		case ast::UnOpType::MARK_MUT:
			paintKw<Tok::MUT>(se, "mut");
			break;

		case ast::UnOpType::PTR:
			paintKw<Tok::PTR>(se, "*");
			break;
		case ast::UnOpType::PTR_CONST:
			paintKw<Tok::PTR_CONST>(se, "*");
			paintKw<Tok::PTR_CONST>(se, "const");
			break;
		case ast::UnOpType::PTR_MUT:
			paintKw<Tok::PTR_MUT>(se, "*");
			paintKw<Tok::PTR_MUT>(se, "mut");
			break;
		case ast::UnOpType::PTR_SHARE:
			paintKw<Tok::PTR_SHARE>(se, "*");
			paintKw<Tok::PTR_SHARE>(se, "share");
			break;

		case ast::UnOpType::REF:
			paintKw<Tok::REF>(se, "&");
			paintLifetime(se, itm.life);
			break;
		case ast::UnOpType::REF_MUT:
			paintKw<Tok::REF_MUT>(se, "&");
			paintLifetime(se, itm.life);
			paintKw<Tok::REF_MUT>(se, "mut");
			break;
		case ast::UnOpType::REF_CONST:
			paintKw<Tok::REF_CONST>(se, "&");
			paintLifetime(se, itm.life);
			paintKw<Tok::REF_CONST>(se, "const");
			break;
		case ast::UnOpType::REF_SHARE:
			paintKw<Tok::REF_SHARE>(se, "&");
			paintLifetime(se, itm.life);
			paintKw<Tok::REF_SHARE>(se, "share");
			break;
		case ast::UnOpType::NONE:
			break;
		}
	}
}