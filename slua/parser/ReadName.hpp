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

#include "SkipSpace.hpp"
#include "RequireToken.hpp"

namespace sluaParse
{
	inline std::string readName(AnyInput auto& in, const bool allowError = false)
	{
		/*
		Names (also called identifiers) in Lua can be any string
		of Latin letters, Arabic-Indic digits, and underscores,
		not beginning with a digit and not being a reserved word.

		The following keywords are reserved and cannot be used as names:

		 and       break     do        else      elseif    end
		 false     for       function  goto      if        in
		 local     nil       not       or        repeat    return
		 then      true      until     while
		*/
		skipSpace(in);

		if (!in)
			throw UnexpectedFileEndError("Expected identifier: but file ended" + errorLocStr(in));


		static const std::unordered_set<std::string> reservedKeywords = {
			"and", "break", "do", "else", "elseif", "end", "false", "for", "function",
			"goto", "if", "in", "local", "nil", "not", "or", "repeat", "return",
			"then", "true", "until", "while"
		};

		const uint8_t firstChar = in.peek();

		// Ensure the first character is valid (a letter or underscore)
		if (!std::isalpha(firstChar) && firstChar != '_')
		{
			if (allowError)
				return "";
			throw UnexpectedCharacterError("Invalid identifier start: must begin with a letter or underscore" + errorLocStr(in));
		}


		std::string res = in.get(); // Consume the first valid character

		// Consume subsequent characters (letters, digits, or underscores)
		while (in)
		{
			const uint8_t ch = in.peek();
			if (std::isalnum(ch) || ch == '_')
			{
				res += in.get();
				continue;
			}
			break; // Stop when a non-identifier character is found
		}

		// Check if the resulting string is a reserved keyword
		if (reservedKeywords.find(res) != reservedKeywords.end())
		{
			if (allowError)
				return "";
			throw ReservedNameError("Invalid identifier: matches a reserved keyword" + errorLocStr(in));
		}

		return res;
	}


	inline std::vector<std::string> readNames(AnyInput auto& in, const bool requires1 = true)
	{
		std::vector<std::string> res;

		if (requires1)
			res.push_back(readName(in));

		bool skipComma = !requires1;//comma wont exist if the first one doesnt exist
		bool allowNameError = !requires1;//if the first one doesnt exist

		while (true)
		{
			if (!skipComma)
			{
				if (!requireTokenNoThrow(in, ","))
					return res;// that must have been the last item
			}
			else
				skipComma = false;// Only skip first comma

			const std::string str = readName(in, allowNameError);

			if (allowNameError && str.empty())
				return {};//no names

			res.push_back(str);

			allowNameError = false;//not the first one anymore
		}

	}

	inline NameList readNameList(AnyInput auto& in)
	{
		/*
			namelist ::= Name {‘,’ Name}
		*/
		NameList ret{};
		ret.push_back(readName(in));

		while (checkReadToken(in, ","))
		{
			ret.push_back(readName(in));
		}
		return ret;
	}
}