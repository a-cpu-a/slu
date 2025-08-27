/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <cstdint>
#include <format>

import slu.ast.state;
import slu.parse.input;
import slu.parse.com.name;
import slu.parse.com.skip_space;
import slu.parse.com.token;

namespace slu::parse
{
	//Doesnt skip space, the current character must be a valid args starter
	template<AnyInput In>
	inline Args readArgs(In& in, const bool allowVarArg)
	{
		const char ch = in.peek();
		if (ch == '"' || ch == '\'' || ch == '[')
		{
			return ArgsType::String(readStringLiteral(in, ch),in.getLoc());
		}
		else if (ch == '(')
		{
			in.skip();//skip start
			skipSpace(in);
			ArgsType::ExprList res{};
			if (in.peek() == ')')// Check if 0 args
			{
				in.skip();
				return res;
			}
			res = readExprList(in, allowVarArg);
			requireToken(in, ")");
			return res;
		}
		else if (ch == '{')
		{
			return ArgsType::Table<In>(readTable(in, allowVarArg));
		}
		throw UnexpectedCharacterError(std::format(
			"Expected function arguments ("
			LUACC_SINGLE_STRING(",")
			" or "
			LUACC_SINGLE_STRING(";")
			"), found " LUACC_SINGLE_STRING("{}")
			"{}"
			, ch, errorLocStr(in)));
	}
}