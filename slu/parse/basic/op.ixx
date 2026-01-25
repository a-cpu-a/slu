/*
    A program file.
    Copyright (C) 2026 a-cpu-a <any1word@proton.me>

    This file is part of Slu-c.

    Slu-c is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Slu-c is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with Slu-c.  If not, see <https://www.gnu.org/licenses/>.

      SPDX-License-Identifier: AGPL3.0-or-later
*/
module;
#include <cstdint>
#include <unordered_set>

export module slu.parse.basic.op;

import slu.ast.enums;
import slu.ast.state;
import slu.parse.input;
import slu.parse.com.skip_space;
import slu.parse.com.tok;

namespace slu::parse
{
	export ast::UnOpType readOptUnOp(AnyInput auto& in)
	{
		skipSpace(in);

		switch (in.peek())
		{
		case '-': in.skip(); return ast::UnOpType::NEG;
		case '!': in.skip(); return ast::UnOpType::NOT;
		case '[':
		{
			const char xCh = in.peekAt(1);
			if (xCh == '=' || xCh == '[') // [=.... // [[....
				break;                    // Its a string
			in.skip();
			requireToken(in, "]");
			return ast::UnOpType::SLICIFY;
		}
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
			if (checkReadTextToken(in, "alloc"))
				return ast::UnOpType::ALLOC;
			break;
		case '.':
			if (checkReadToken<false>(in, ".."))
				return ast::UnOpType::RANGE_BEFORE;
			break;
		case 'm':
			if (checkReadTextToken(in, "mut"))
				return ast::UnOpType::MARK_MUT;
			break;
		default: break;
		}
		return ast::UnOpType::NONE;
	}
	export template<bool noRangeOp>
	ast::PostUnOpType readOptPostUnOp(AnyInput auto& in)
	{
		skipSpace(in);

		if (in)
		{
			switch (in.peek())
			{
			case '?': in.skip(); return ast::PostUnOpType::TRY;
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

	export ast::BinOpType readOptBinOp(AnyInput auto& in)
	{
		switch (in.peek())
		{
		case '+':
			in.skip();
			if (checkReadToken<false>(in, "+")) //++
				return ast::BinOpType::CONCAT;
			return ast::BinOpType::ADD;
		case '-': in.skip(); return ast::BinOpType::SUB;
		case '*':
			in.skip();
			if (checkReadToken<false>(in, "*")) //**
				return ast::BinOpType::REP;
			return ast::BinOpType::MUL;
		case '/':
			in.skip();
			if (checkReadToken<false>(in, "/")) // '//'
				return ast::BinOpType::FLOOR_DIV;
			return ast::BinOpType::DIV;
		case '^': in.skip(); return ast::BinOpType::EXP;
		case '%': in.skip(); return ast::BinOpType::REM;
		case '&': in.skip(); return ast::BinOpType::BIT_AND;
		case '!':
			in.skip();
			requireToken<false>(in, "=");
			return ast::BinOpType::NE;
			break;
		case '~':
			in.skip();
			if (checkReadToken<false>(in, "~")) //~~
				return ast::BinOpType::MK_RESULT;
			return ast::BinOpType::BIT_XOR;
		case '|':
			in.skip();
			if (checkReadToken<false>(in, "|")) //||
				return ast::BinOpType::UNION;
			return ast::BinOpType::BIT_OR;
		case '>':
			in.skip();
			if (checkReadToken<false>(in, ">")) //>>
				return ast::BinOpType::SHR;
			if (checkReadToken<false>(in, "=")) //>=
				return ast::BinOpType::GE;
			return ast::BinOpType::GT;
		case '<':
			in.skip();
			if (checkReadToken<false>(in, "<")) //<<
				return ast::BinOpType::SHL;
			if (checkReadToken<false>(in, "=")) //<=
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
} //namespace slu::parse