/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <cstdint>
#include <unordered_set>

//https://www.lua.org/manual/5.4/manual.html
//https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form
//https://www.sciencedirect.com/topics/computer-science/backus-naur-form

import slu.ast.enums;
import slu.ast.state;
import slu.parse.input;
import slu.parse.com.skip_space;
#include <slu/parse/adv/RequireToken.hpp>

namespace slu::parse
{
	inline ast::UnOpType readOptUnOp(AnyInput auto& in,const bool typeOnly=false)
	{
		skipSpace(in);

		switch (in.peek())
		{
		case '-':
			if (typeOnly)break;
			in.skip();
			return ast::UnOpType::NEG;
		case '!':
			if (typeOnly)break;
			//if (in.peekAt(1) == '=')
			//	break;//Its !=
			in.skip();
			return ast::UnOpType::NOT;
			break;
		case '&':
			in.skip();
			if (checkReadTextToken(in, "mut"))
				return ast::UnOpType::REF_MUT;
			if (checkReadTextToken(in, "const"))
				return ast::UnOpType::REF_CONST;
			if (checkReadTextToken(in, "share"))
				return ast::UnOpType::REF_SHARE;
			return ast::UnOpType::REF;
		case '*':
			in.skip();
			if (checkReadTextToken(in, "mut"))
				return ast::UnOpType::PTR_MUT;
			if (checkReadTextToken(in, "const"))
				return ast::UnOpType::PTR_CONST;
			if (checkReadTextToken(in, "share"))
				return ast::UnOpType::PTR_SHARE;

			return ast::UnOpType::PTR;
		case 'a':
			if (typeOnly)break;

			if (checkReadTextToken(in, "alloc"))
				return ast::UnOpType::ALLOC;
			break;
		case '.':
			if (typeOnly)break;
			if (checkReadToken<false>(in, ".."))
				return ast::UnOpType::RANGE_BEFORE;
			break;
		case 'm':
			if (checkReadTextToken(in, "mut"))
				return ast::UnOpType::MARK_MUT;
			break;
		default:
			break;
		}
		return ast::UnOpType::NONE;
	}
	template<bool noRangeOp>
	inline ast::PostUnOpType readOptPostUnOp(AnyInput auto& in)
	{
		skipSpace(in);

		if(in)
		{
			switch (in.peek())
			{
			case '?':
				in.skip();
				return ast::PostUnOpType::TRY;
			case '.':
				if (checkReadToken<false>(in, ".*"))
					return ast::PostUnOpType::DEREF;
				if constexpr (!noRangeOp)
				{
					if (checkReadToken<false>(in, ".."))
						return ast::PostUnOpType::RANGE_AFTER;
				}
				break;
			}
		}
		return ast::PostUnOpType::NONE;
	}

	inline ast::BinOpType readOptBinOp(AnyInput auto& in)
	{
		switch (in.peek())
		{
		case '+':
			in.skip();
			if (checkReadToken<false>(in, "+"))//++
				return ast::BinOpType::CONCAT;
			return ast::BinOpType::ADD;
		case '-':
			in.skip();
			return ast::BinOpType::SUB;
		case '*':
			in.skip();
			if (checkReadToken<false>(in, "*"))//**
				return ast::BinOpType::REP;
			return ast::BinOpType::MUL;
		case '/':
			in.skip();
			if (checkReadToken<false>(in, "/"))// '//'
				return ast::BinOpType::FLOOR_DIV;
			return ast::BinOpType::DIV;
		case '^':
			in.skip();
			return ast::BinOpType::EXP;
		case '%':
			in.skip();
			return ast::BinOpType::REM;
		case '&':
			in.skip();
			return ast::BinOpType::BIT_AND;
		case '!':
			in.skip();
			requireToken<false>(in, "=");
			return ast::BinOpType::NE;
			break;
		case '~':
			in.skip();
			if (checkReadToken<false>(in, "~"))//~~
				return ast::BinOpType::MK_RESULT;
			return ast::BinOpType::BIT_XOR;
		case '|':
			in.skip();
			if (checkReadToken<false>(in, "|"))//||
				return ast::BinOpType::UNION;
			return ast::BinOpType::BIT_OR;
		case '>':
			in.skip();
			if (checkReadToken<false>(in, ">"))//>>
				return ast::BinOpType::SHR;
			if (checkReadToken<false>(in, "="))//>=
				return ast::BinOpType::GE;
			return ast::BinOpType::GT;
		case '<':
			in.skip();
			if (checkReadToken<false>(in, "<"))//<<
				return ast::BinOpType::SHL;
			if (checkReadToken<false>(in, "="))//<=
				return ast::BinOpType::LE;
			return ast::BinOpType::LT;
		case '=':
			if (checkReadToken<false>(in, "=="))
				return ast::BinOpType::EQ;
			break;
		case 'a':
			if (checkReadTextToken(in, "and"))
				return ast::BinOpType::AND;
			if (checkReadTextToken(in, "as"))
				return ast::BinOpType::AS;
			break;
		case 'o':
			if (checkReadTextToken(in, "or"))
				return ast::BinOpType::OR;
			break;
		case '.':
			if (checkReadToken<false>(in, ".."))
				return ast::BinOpType::RANGE;
			break;
		}
		return ast::BinOpType::NONE;
	}
}