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
	template<AnyInput In>
	inline Field<In> readField(In& in, const bool allowVarArg)
	{
		// field :: = ‘[’ exp ‘]’ ‘ = ’ exp | Name ‘ = ’ exp | exp
		skipSpace(in);

		if (checkReadToken(in, "["))
		{
			FieldType::EXPR2EXPR<In> res{};

			res.idx = readExpr(in,allowVarArg);

			requireToken(in, "]");
			requireToken(in, "=");

			res.v = readExpr(in,allowVarArg);

			return res;
		}

		const size_t nameOffset = peekName(in);

		if (nameOffset !=SIZE_MAX)
		{
			//Lazy-TODO: eof handling lol
			const size_t spacedOffset = weakSkipSpace(in, nameOffset);

			//check at the CORRECT position, AND that it ISNT ==
			if (in.peekAt(spacedOffset)=='=' && in.peekAt(spacedOffset+1)!='=')
			{
				std::string name = readName(in);
				skipSpace(in);
				in.skip();// '='

				return FieldType::NAME2EXPR<In>(name, readExpr(in,allowVarArg));
			}
		}

		return FieldType::EXPR<In>(readExpr(in,allowVarArg));
	}

	//Will NOT check the first char '{' !!!
	//But will skip it
	template<AnyInput In>
	inline TableConstructor<In> readTableConstructor(In& in, const bool allowVarArg)
	{
		/*
			tableconstructor ::= ‘{’ [fieldlist] ‘}’
			fieldlist ::= field {fieldsep field} [fieldsep]
			fieldsep ::= ‘,’ | ‘;’
		*/

		in.skip();//get rid of '{'

		skipSpace(in);

		TableConstructor<In> tbl{};

		if (in.peek() == '}')
		{
			in.skip();
			return tbl;
		}
		//must be field
		tbl.emplace_back(readField(in,allowVarArg));

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
					"), found " LUACC_SINGLE_STRING("{}")
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
			tbl.emplace_back(readField(in,allowVarArg));
		}
		return tbl;
	}
}