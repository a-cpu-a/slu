/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <cstdint>
#include <unordered_set>

//https://www.lua.org/manual/5.4/manual.html
//https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form
//https://www.sciencedirect.com/topics/computer-science/backus-naur-form

import slu.ast.state;
import slu.parse.input;
#include <slu/parse/adv/SkipSpace.hpp>
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
			return ast::UnOpType::NEGATE;
		case '!':
			if (typeOnly)break;
			//if (in.peekAt(1) == '=')
			//	break;//Its !=
			in.skip();
			return ast::UnOpType::LOGICAL_NOT;
			break;
		case '&':
			in.skip();
			if (checkReadTextToken(in, "mut"))
				return ast::UnOpType::TO_REF_MUT;
			if (checkReadTextToken(in, "const"))
				return ast::UnOpType::TO_REF_CONST;
			if (checkReadTextToken(in, "share"))
				return ast::UnOpType::TO_REF_SHARE;
			return ast::UnOpType::TO_REF;
		case '*':
			in.skip();
			if (checkReadTextToken(in, "mut"))
				return ast::UnOpType::TO_PTR_MUT;
			if (checkReadTextToken(in, "const"))
				return ast::UnOpType::TO_PTR_CONST;
			if (checkReadTextToken(in, "share"))
				return ast::UnOpType::TO_PTR_SHARE;

			return ast::UnOpType::TO_PTR;
		case 'a':
			if (typeOnly)break;

			if (checkReadTextToken(in, "alloc"))
				return ast::UnOpType::ALLOCATE;
			break;
		case '.':
			if (typeOnly)break;
			if (checkReadToken<false>(in, ".."))
				return ast::UnOpType::RANGE_BEFORE;
			break;
		case 'm':
			if (checkReadTextToken(in, "mut"))
				return ast::UnOpType::MUT;
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
				return ast::PostUnOpType::PROPOGATE_ERR;
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
				return ast::BinOpType::CONCATENATE;
			return ast::BinOpType::ADD;
		case '-':
			in.skip();
			return ast::BinOpType::SUBTRACT;
		case '*':
			in.skip();
			if (checkReadToken<false>(in, "*"))//**
				return ast::BinOpType::ARRAY_MUL;
			return ast::BinOpType::MULTIPLY;
		case '/':
			in.skip();
			if (checkReadToken<false>(in, "/"))// '//'
				return ast::BinOpType::FLOOR_DIVIDE;
			return ast::BinOpType::DIVIDE;
		case '^':
			in.skip();
			return ast::BinOpType::EXPONENT;
		case '%':
			in.skip();
			return ast::BinOpType::MODULO;
		case '&':
			in.skip();
			return ast::BinOpType::BITWISE_AND;
		case '!':
			in.skip();
			requireToken<false>(in, "=");
			return ast::BinOpType::NOT_EQUAL;
			break;
		case '~':
			in.skip();
			if (checkReadToken<false>(in, "~"))//~~
				return ast::BinOpType::MAKE_RESULT;
			return ast::BinOpType::BITWISE_XOR;
		case '|':
			in.skip();
			if (checkReadToken<false>(in, "|"))//||
				return ast::BinOpType::UNION;
			return ast::BinOpType::BITWISE_OR;
		case '>':
			in.skip();
			if (checkReadToken<false>(in, ">"))//>>
				return ast::BinOpType::SHIFT_RIGHT;
			if (checkReadToken<false>(in, "="))//>=
				return ast::BinOpType::GREATER_EQUAL;
			return ast::BinOpType::GREATER_THAN;
		case '<':
			in.skip();
			if (checkReadToken<false>(in, "<"))//<<
				return ast::BinOpType::SHIFT_LEFT;
			if (checkReadToken<false>(in, "="))//<=
				return ast::BinOpType::LESS_EQUAL;
			return ast::BinOpType::LESS_THAN;
		case '=':
			if (checkReadToken<false>(in, "=="))
				return ast::BinOpType::EQUAL;
			break;
		case 'a':
			if (checkReadTextToken(in, "and"))
				return ast::BinOpType::LOGICAL_AND;
			if (checkReadTextToken(in, "as"))
				return ast::BinOpType::AS;
			break;
		case 'o':
			if (checkReadTextToken(in, "or"))
				return ast::BinOpType::LOGICAL_OR;
			break;
		case '.':
			if (checkReadToken<false>(in, ".."))
				return ast::BinOpType::RANGE_BETWEEN;
			break;
		}
		return ast::BinOpType::NONE;
	}
}