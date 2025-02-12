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
#include <slua/parser/adv/ReadExpr.hpp>

namespace sluaParse
{
	constexpr bool isFieldSep(const char ch)
	{
		return ch == ',' || ch == ';';
	}
	inline Field readField(AnyInput auto& in)
	{
		// field :: = ‘[’ exp ‘]’ ‘ = ’ exp | Name ‘ = ’ exp | exp
		skipSpace(in);

		if (checkReadToken(in, "["))
		{
			FieldType::EXPR2EXPR res{};

			res.idx = readExpr(in);

			requireToken(in, "]");
			requireToken(in, "=");

			res.v = readExpr(in);

			return res;
		}

		std::string name = peekName(in);

		if (!name.empty())
		{
			//TODO: check at the CORRECT position... AND ISNT ==, so...
			if (checkReadToken(in, "="))
			{
				return FieldType::NAME2EXPR(name, readExpr(in));
			}
		}

		return FieldType::EXPR(readExpr(in));
	}

	//Will NOT check the first char '{' !!!
	//But will skip it
	inline TableConstructor readTableConstructor(AnyInput auto& in)
	{
		/*
			tableconstructor ::= ‘{’ [fieldlist] ‘}’
			fieldlist ::= field {fieldsep field} [fieldsep]
			fieldsep ::= ‘,’ | ‘;’
		*/

		in.skip();//get rid of '{'

		skipSpace(in);

		TableConstructor tbl{};

		if (in.peek() == '}')
		{
			in.skip();
			return tbl;
		}
		//must be field
		tbl.emplace_back(readField(in));

		while (true)
		{
			skipSpace(in);
			const char ch = in.peek();
			if (ch == '}')
			{
				in.skip();
				break;
			}
			if (!isFieldSep(ch))
			{
				throw UnexpectedCharacterError(std::format(
					"Expected table separator ("
					LUACC_SINGLE_STRING(",")
					" or "
					LUACC_SINGLE_STRING(";")
					"), found " LUACC_START_SINGLE_STRING "{}" LUACC_END_SINGLE_STRING
					"{}"
				, ch, errorLocStr(in)));
			}
			in.skip();//skip field-sep

			skipSpace(in);
			if (in.peek() == '}')
			{
				in.skip();
				break;
			}
			tbl.emplace_back(readField(in));
		}
		return tbl;
	}
}