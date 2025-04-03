/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <cstdint>
#include <unordered_set>

//https://www.lua.org/manual/5.4/manual.html
//https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form
//https://www.sciencedirect.com/topics/computer-science/backus-naur-form

#include <slua/parser/State.hpp>
#include <slua/parser/Input.hpp>
#include <slua/parser/adv/SkipSpace.hpp>
#include <slua/parser/adv/RequireToken.hpp>

namespace sluaParse
{
	inline UnOpType readOptUnOp(AnyInput auto& in)
	{
		skipSpace(in);

		switch (in.peek())
		{
		case '-':
			in.skip();
			return UnOpType::NEGATE;
		case 'n':
			if (checkReadTextToken(in, "not"))
				return UnOpType::LOGICAL_NOT;
			break;
		case '#':
			in.skip();
			return UnOpType::LENGTH;
		case '~':
			in.skip();
			return UnOpType::BITWISE_NOT;
		case '&':
			if (!(in.settings() & sluaSyn))break;

			in.skip();
			if (checkReadTextToken(in, "mut"))
				return UnOpType::TO_REF_MUT;
			return UnOpType::TO_REF;
		case '*':
			if (!(in.settings() & sluaSyn))break;

			in.skip();
			if (checkReadTextToken(in, "mut"))
				return UnOpType::TO_PTR_MUT;
			if (checkReadTextToken(in, "const"))
				return UnOpType::TO_PTR_CONST;

			return UnOpType::DEREF;
		default:
			break;
		}
		return UnOpType::NONE;
	}
	inline PostUnOpType readOptPostUnOp(AnyInput auto& in)
	{
		skipSpace(in);

		switch (in.peek())
		{
		case '?':
			in.skip();
			return PostUnOpType::PROPOGATE_ERR;
		}
		return PostUnOpType::NONE;
	}

	inline BinOpType readOptBinOp(AnyInput auto& in)
	{
		switch (in.peek())
		{
		case '+':
			in.skip();
			return BinOpType::ADD;
		case '-':
			in.skip();
			return BinOpType::SUBTRACT;
		case '*':
			in.skip();
			return BinOpType::MULTIPLY;
		case '/':
			in.skip();
			if (checkReadToken(in, "/"))// '//'
				return BinOpType::FLOOR_DIVIDE;
			return BinOpType::DIVIDE;
		case '^':
			in.skip();
			return BinOpType::EXPONENT;
		case '%':
			in.skip();
			return BinOpType::MODULO;
		case '&':
			in.skip();
			return BinOpType::BITWISE_AND;
		case '~':
			in.skip();
			if (checkReadToken(in, "="))//~=
				return BinOpType::NOT_EQUAL;
			return BinOpType::BITWISE_XOR;
		case '|':
			in.skip();
			return BinOpType::BITWISE_OR;
		case '>':
			in.skip();
			if (checkReadToken(in, ">"))//>>
				return BinOpType::SHIFT_RIGHT;
			if (checkReadToken(in, "="))//>=
				return BinOpType::GREATER_EQUAL;
			return BinOpType::GREATER_THAN;
		case '<':
			in.skip();
			if (checkReadToken(in, "<"))//<<
				return BinOpType::SHIFT_LEFT;
			if (checkReadToken(in, "="))//<=
				return BinOpType::LESS_EQUAL;
			return BinOpType::LESS_THAN;
		case '=':
			if (checkReadToken(in, "=="))
				return BinOpType::EQUAL;
			break;
		case '.':
			if (checkReadToken(in, ".."))
				return BinOpType::CONCATENATE;
			break;
		case 'a':
			if (checkReadTextToken(in, "and"))
				return BinOpType::LOGICAL_AND;
			break;
		case 'o':
			if (checkReadTextToken(in, "or"))
				return BinOpType::LOGICAL_OR;
			break;
		}
		return BinOpType::NONE;
	}
}