/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <cstdint>

#include <slu/parser/State.hpp>
#include <slu/parser/Input.hpp>
#include <slu/parser/adv/SkipSpace.hpp>
#include <slu/parser/adv/RequireToken.hpp>
#include <slu/parser/adv/ReadName.hpp>

namespace slu::parse
{
	//Doesnt skip space, the current character must be a valid args starter
	template<AnyInput In>
	inline Args<In> readArgs(In& in, const bool allowVarArg)
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
			ArgsType::ExprList<In> res{};
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